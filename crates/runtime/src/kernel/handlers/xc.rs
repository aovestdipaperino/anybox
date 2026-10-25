//! The kernel's cryptography exports: SHA-1 and HMAC-SHA-1, which titles sign their saves with;
//! RC4; DES and two-key triple DES; and RSA on the console's key layout.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission, and adapted
//! to this crate's handler interface.

use hmac::{Hmac, Mac};
use num_bigint::BigUint;

use super::value;
use crate::kernel::{Call, CallResult, KResult, Kernel};

/// SHA-1's starting state.
const SHA_START: [u32; 5] = [
    0x6745_2301,
    0xEFCD_AB89,
    0x98BA_DCFE,
    0x1032_5476,
    0xC3D2_E1F0,
];
const BLOCK: usize = 64;
/// The title's SHA context, which it only passes back: the five state words, the bytes hashed
/// so far (64 bits), and the partial block. 92 bytes, within the console's 116-byte context.
const STATE: u32 = 0;
const LENGTH: u32 = 20;
const PARTIAL: u32 = 28;

/// A SHA-1 computation kept in the title's context, so it can be resumed at each call.
struct Sha {
    state: [u32; 5],
    length: u64,
    partial: Vec<u8>,
}

impl Sha {
    fn new() -> Self {
        Self {
            state: SHA_START,
            length: 0,
            partial: Vec::new(),
        }
    }

    fn load(k: &Kernel, at: u32) -> KResult<Self> {
        let mut state = [0; 5];
        for (index, word) in (0u32..).zip(state.iter_mut()) {
            *word = k.r32(at + STATE + index * 4)?;
        }
        let length = k.r64(at + LENGTH)?;
        let partial = k.read(at + PARTIAL, (length % BLOCK as u64) as u32)?;
        Ok(Self {
            state,
            length,
            partial,
        })
    }

    fn store(&self, k: &Kernel, at: u32) -> KResult<()> {
        for (index, word) in (0u32..).zip(self.state) {
            k.w32(at + STATE + index * 4, word)?;
        }
        k.w64(at + LENGTH, self.length)?;
        k.write(at + PARTIAL, &self.partial)
    }

    fn update(&mut self, data: &[u8]) {
        self.length += data.len() as u64;
        self.partial.extend_from_slice(data);
        let whole = self.partial.len() / BLOCK * BLOCK;
        let blocks: Vec<_> = self.partial[..whole]
            .as_chunks::<BLOCK>()
            .0
            .iter()
            .map(|block| (*block).into())
            .collect();
        sha1::compress(&mut self.state, &blocks);
        self.partial.drain(..whole);
    }

    fn finish(mut self) -> [u8; 20] {
        let bits = self.length.wrapping_mul(8);
        let mut padding = vec![0x80];
        padding.resize((BLOCK + 55 - self.partial.len()) % BLOCK + 1, 0);
        padding.extend_from_slice(&bits.to_be_bytes());
        self.update(&padding);
        let mut digest = [0; 20];
        for (bytes, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
            *bytes = word.to_be_bytes();
        }
        digest
    }
}

/// `XcSHAInit(context)`.
pub fn sha_init(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    Sha::new().store(k, c.arg(0))?;
    value(0)
}

/// `XcSHAUpdate(context, input, length)`.
pub fn sha_update(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (at, input, len) = (c.arg(0), c.arg(1), c.arg(2));
    let mut sha = Sha::load(k, at)?;
    let data = k.read(input, len)?;
    sha.update(&data);
    sha.store(k, at)?;
    value(0)
}

/// `XcSHAFinal(context, digest)`: the 20-byte digest.
pub fn sha_final(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let digest = Sha::load(k, c.arg(0))?.finish();
    k.write(c.arg(1), &digest)?;
    value(0)
}

/// `XcHMAC(key, key_length, data, data_length, data2, data2_length, digest)`: HMAC-SHA-1 of the
/// two inputs one after the other; the second may be absent.
pub fn hmac(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    // The console's HMAC uses at most 64 bytes of key, where the standard one hashes a longer
    // key first.
    let key = k.read(c.arg(0), c.arg(1).min(64))?;
    let Ok(mut mac) = Hmac::<sha1::Sha1>::new_from_slice(&key) else {
        return Ok(CallResult::Unsupported(
            "an HMAC key it would not take".to_owned(),
        ));
    };
    for (data, len) in [(c.arg(2), c.arg(3)), (c.arg(4), c.arg(5))] {
        if data != 0 && len != 0 {
            mac.update(&k.read(data, len)?);
        }
    }
    k.write(c.arg(6), &mac.finalize().into_bytes())?;
    value(0)
}

/// An RC4 state in the title's key structure: the permutation, then the two indices.
const RC4_I: u32 = 256;
const RC4_J: u32 = 257;

/// `XcRC4Key(KeyStruct, KeyLength, Key)`: the key schedule.
pub fn rc4_key(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (state, key) = (c.arg(0), k.read(c.arg(2), c.arg(1))?);
    let mut s: Vec<u8> = (0..=255).collect();
    if !key.is_empty() {
        let mut j = 0u8;
        for i in 0..256 {
            j = j.wrapping_add(s[i]).wrapping_add(key[i % key.len()]);
            s.swap(i, usize::from(j));
        }
    }
    k.write(state, &s)?;
    k.write(state + RC4_I, &[0, 0])?;
    value(0)
}

/// `XcRC4Crypt(KeyStruct, InputLength, Input)`: encrypts or decrypts in place.
pub fn rc4_crypt(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (state, len, data_at) = (c.arg(0), c.arg(1), c.arg(2));
    let mut perm = k.read(state, 256)?;
    let (mut i, mut j) = (k.r8(state + RC4_I)?, k.r8(state + RC4_J)?);
    let mut data = k.read(data_at, len)?;
    for byte in &mut data {
        i = i.wrapping_add(1);
        j = j.wrapping_add(perm[usize::from(i)]);
        perm.swap(usize::from(i), usize::from(j));
        *byte ^= perm[usize::from(perm[usize::from(i)].wrapping_add(perm[usize::from(j)]))];
    }
    k.write(data_at, &data)?;
    k.write(state, &perm)?;
    k.write(state + RC4_I, &[i, j])?;
    value(0)
}

/// `XcDESKeyParity(Key, KeyLength)`: gives each byte odd parity in its low bit.
pub fn des_key_parity(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let mut key = k.read(c.arg(0), c.arg(1))?;
    for byte in &mut key {
        *byte = (*byte & 0xFE) | u8::from((*byte >> 1).count_ones() % 2 == 0);
    }
    k.write(c.arg(0), &key)?;
    value(0)
}

/// The cipher a key table is for: DES (0) with an 8-byte key, or triple DES (else) with a
/// 16-byte one, its third key the first. The table holds the key itself.
fn key_len(cipher: u32) -> u32 {
    if cipher == 0 { 8 } else { 16 }
}

/// `XcKeyTable(Cipher, KeyTable, Key)`.
pub fn key_table(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let key = k.read(c.arg(2), key_len(c.arg(0)))?;
    k.write(c.arg(1), &key)?;
    value(0)
}

/// Encrypts (`encrypt`) or decrypts one 8-byte block with `key`; `None` for a key of the wrong
/// length.
fn des_block(key: &[u8], cipher: u32, block: &mut [u8; 8], encrypt: bool) -> Option<()> {
    use des::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
    let block = des::cipher::generic_array::GenericArray::from_mut_slice(block);
    if cipher == 0 {
        let des = des::Des::new_from_slice(key).ok()?;
        if encrypt {
            des.encrypt_block(block);
        } else {
            des.decrypt_block(block);
        }
    } else {
        let des = des::TdesEde2::new_from_slice(key).ok()?;
        if encrypt {
            des.encrypt_block(block);
        } else {
            des.decrypt_block(block);
        }
    }
    Some(())
}

fn bad_des_key() -> KResult<CallResult> {
    Ok(CallResult::Unsupported(
        "a DES key of the wrong length".to_owned(),
    ))
}

/// Reads an 8-byte block.
fn read_block(k: &Kernel, at: u32) -> KResult<[u8; 8]> {
    let mut block = [0; 8];
    k.read_into(at, &mut block)?;
    Ok(block)
}

/// `XcBlockCrypt(Cipher, Output, Input, KeyTable, Op)`: one block; `Op` 1 encrypts, 0 decrypts.
pub fn block_crypt(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let cipher = c.arg(0);
    let key = k.read(c.arg(3), key_len(cipher))?;
    let mut block = read_block(k, c.arg(2))?;
    if des_block(&key, cipher, &mut block, c.arg(4) != 0).is_none() {
        return bad_des_key();
    }
    k.write(c.arg(1), &block)?;
    value(0)
}

/// `XcBlockCryptCBC(Cipher, InputLength, Output, Input, KeyTable, Op, Feedback)`: cipher block
/// chaining over whole blocks, with the feedback block updated for the next call.
pub fn block_crypt_cbc(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (cipher, len, output, input, table, encrypt, feedback_at) = (
        c.arg(0),
        c.arg(1) / 8 * 8,
        c.arg(2),
        c.arg(3),
        c.arg(4),
        c.arg(5) != 0,
        c.arg(6),
    );
    let key = k.read(table, key_len(cipher))?;
    let mut feedback = read_block(k, feedback_at)?;
    let data = k.read(input, len)?;
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.as_chunks::<8>().0 {
        let mut block = *chunk;
        if encrypt {
            for (byte, previous) in block.iter_mut().zip(feedback) {
                *byte ^= previous;
            }
            if des_block(&key, cipher, &mut block, true).is_none() {
                return bad_des_key();
            }
            feedback = block;
        } else {
            if des_block(&key, cipher, &mut block, false).is_none() {
                return bad_des_key();
            }
            for (byte, previous) in block.iter_mut().zip(feedback) {
                *byte ^= previous;
            }
            feedback = *chunk;
        }
        out.extend_from_slice(&block);
    }
    k.write(output, &out)?;
    k.write(feedback_at, &feedback)?;
    value(0)
}

/// Writes `number` as `len` bytes, least significant first, at `at`.
fn write_number(k: &Kernel, at: u32, number: &BigUint, len: usize) -> KResult<()> {
    let bytes = number.to_bytes_le();
    let mut out = vec![0; len];
    let n = bytes.len().min(len);
    out[..n].copy_from_slice(&bytes[..n]);
    k.write(at, &out)
}

/// `XcModExp(A, B, C, D, N)`: A = B to the power C, modulo D; all `N` dwords, least significant
/// first. Returns whether it could be computed.
pub fn mod_exp(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let len = c.arg(4).wrapping_mul(4);
    let read = |at: u32| -> KResult<BigUint> { Ok(BigUint::from_bytes_le(&k.read(at, len)?)) };
    let (base, exponent, modulus) = (read(c.arg(1))?, read(c.arg(2))?, read(c.arg(3))?);
    if modulus == BigUint::ZERO {
        return value(0);
    }
    write_number(k, c.arg(0), &base.modpow(&exponent, &modulus), len as usize)?;
    value(1)
}

/// An RSA key in the console's layout: `RSA1` (public) or `RSA2` (private), the modulus buffer's
/// length in bytes, the modulus's bits, the most data it signs, the public exponent, then the
/// modulus, least significant byte first. A private key follows it with the two primes, the
/// two exponents and the coefficient (each half as long), then the private exponent.
struct Key {
    len: u32,
    modulus: BigUint,
    exponent: BigUint,
}

const KEY_HEADER: u32 = 20;

fn read_key(k: &Kernel, at: u32, private: bool) -> KResult<Option<Key>> {
    let magic = k.read(at, 4)?;
    let wanted: &[u8] = if private { b"RSA2" } else { b"RSA1" };
    if magic != wanted {
        return Ok(None);
    }
    let len = k.r32(at + 4)?;
    let modulus = BigUint::from_bytes_le(&k.read(at + KEY_HEADER, len)?);
    let exponent = if private {
        let offset = KEY_HEADER
            .wrapping_add(len)
            .wrapping_add(5u32.wrapping_mul(len / 2));
        BigUint::from_bytes_le(&k.read(at.wrapping_add(offset), len)?)
    } else {
        BigUint::from(k.r32(at + 16)?)
    };
    Ok(Some(Key {
        len,
        modulus,
        exponent,
    }))
}

/// Raises the `key.len`-byte number at `input` to the key's exponent, into `output`; returns
/// whether it could.
fn rsa(k: &Kernel, key: &Key, input: u32, output: u32) -> KResult<bool> {
    if key.modulus == BigUint::ZERO {
        return Ok(false);
    }
    let number = BigUint::from_bytes_le(&k.read(input, key.len)?);
    write_number(
        k,
        output,
        &number.modpow(&key.exponent, &key.modulus),
        key.len as usize,
    )?;
    Ok(true)
}

/// `XcPKGetKeyLen(PubKey)`: the key's length in bytes.
pub fn pk_get_key_len(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(k.r32(c.arg(0) + 4)?)
}

/// `XcPKEncPublic(PubKey, Input, Output)`.
pub fn pk_enc_public(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(key) = read_key(k, c.arg(0), false)? else {
        return value(0);
    };
    value(u32::from(rsa(k, &key, c.arg(1), c.arg(2))?))
}

/// `XcPKDecPrivate(PrvKey, Input, Output)`.
pub fn pk_dec_private(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(key) = read_key(k, c.arg(0), true)? else {
        return value(0);
    };
    value(u32::from(rsa(k, &key, c.arg(1), c.arg(2))?))
}

/// The `DigestInfo` prefixes the console's kernel accepts after the digest, least significant
/// byte first (`shaEncodings` in the kernel's `xcbase.c`): SHA-1 with a NULL parameter, then
/// without. A block with neither goes straight to its zero separator.
const SHA_ENCODINGS: [&[u8]; 2] = [
    &[
        0x14, 0x04, 0x00, 0x05, 0x1A, 0x02, 0x03, 0x0E, 0x2B, 0x05, 0x06, 0x09, 0x30, 0x21, 0x30,
    ],
    &[
        0x14, 0x04, 0x1A, 0x02, 0x03, 0x0E, 0x2B, 0x05, 0x06, 0x07, 0x30, 0x1F, 0x30,
    ],
];

/// Whether `block`, an RSA public operation's result least significant byte first, is a PKCS #1
/// type 1 block for `digest`, as the kernel's `XCVerifyPKCS1SigningFmt` checks it: the digest
/// reversed, an optional `DigestInfo` prefix, a zero, 0xFF padding, a 1 at `datalen - 1` and a
/// zero at `datalen`.
fn pkcs1_block_matches(block: &[u8], datalen: usize, digest: &[u8]) -> bool {
    if digest.len() != 20 || datalen < 21 || block.len() <= datalen {
        return false;
    }
    if !block.iter().zip(digest.iter().rev()).all(|(a, b)| a == b) {
        return false;
    }
    let mut end = digest.len();
    if let Some(prefix) = SHA_ENCODINGS
        .iter()
        .find(|prefix| block[end..].starts_with(prefix))
    {
        end += prefix.len();
    }
    block.get(end) == Some(&0)
        && block[datalen] == 0
        && block[datalen - 1] == 1
        && block
            .get(end + 1..datalen - 1)
            .is_none_or(|padding| padding.iter().all(|&byte| byte == 0xFF))
}

/// `XcVerifyPKCS1Signature(Sig, PubKey, Digest)`: whether the signature, opened with the public
/// key, is the PKCS #1 block for the 20-byte digest. The console's numbers are little-endian,
/// so the block holds the digest reversed at its start (see [`pkcs1_block_matches`]); the
/// signature is `(bitlen + 7) / 8` bytes and must be below the modulus.
pub fn verify_pkcs1_signature(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(key) = read_key(k, c.arg(1), false)? else {
        return value(0);
    };
    let bit_length = k.r32(c.arg(1) + 8)?;
    let data_length = k.r32(c.arg(1) + 12)? as usize;
    let signature_bytes = bit_length.div_ceil(8);
    if key.modulus == BigUint::ZERO || signature_bytes > key.len {
        return value(0);
    }
    let digest = k.read(c.arg(2), 20)?;
    let signature = BigUint::from_bytes_le(&k.read(c.arg(0), signature_bytes)?);
    if signature >= key.modulus {
        return value(0);
    }
    let mut block = signature.modpow(&key.exponent, &key.modulus).to_bytes_le();
    block.resize(key.len as usize, 0);
    value(u32::from(pkcs1_block_matches(&block, data_length, &digest)))
}

/// `XcCryptService(Operation, Args)`: the console's own does nothing and returns zero.
pub fn crypt_service(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(0)
}

/// `XcUpdateCrypto(NewVector, RomVector)`: a title may replace the kernel's cryptography with
/// its own routines; the kernel's are kept here, which compute the same results.
pub fn update_crypto(_: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::handlers::lookup;
    use crate::kernel::testing::kernel;

    /// Guest scratch memory inside the committed heap.
    const SCRATCH: u32 = crate::mem::HEAP_BASE + 0x2F_0000;

    fn call(k: &mut Kernel, name: &str, args: &[u32]) -> u32 {
        let handler = lookup(name).unwrap_or_else(|| panic!("no handler for {name}"));
        let result = handler(
            k,
            &Call {
                args: args.to_vec(),
                esp: 0,
            },
        );
        match result {
            Ok(CallResult::Value(v)) => v,
            other => panic!("{name} returned {other:?}"),
        }
    }

    fn hex(bytes: &[u8]) -> String {
        use std::fmt::Write as _;
        bytes.iter().fold(String::new(), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
    }

    #[test]
    fn sha_matches_the_standard_vectors_in_any_pieces() {
        assert_eq!(
            hex(&Sha::new().finish()),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        let mut sha = Sha::new();
        sha.update(b"abc");
        assert_eq!(
            hex(&sha.finish()),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        // 56 bytes: the padding needs a second block.
        let text = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let mut sha = Sha::new();
        for piece in text.chunks(7) {
            sha.update(piece);
        }
        assert_eq!(
            hex(&sha.finish()),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
        let mut sha = Sha::new();
        for _ in 0..1000 {
            sha.update(&[b'a'; 1000]);
        }
        assert_eq!(
            hex(&sha.finish()),
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
        );
    }

    #[test]
    fn sha_exports_keep_the_state_in_the_guest_context() {
        let mut k = kernel();
        let (context, input, digest) = (SCRATCH, SCRATCH + 0x100, SCRATCH + 0x200);
        k.write(input, b"abc").unwrap();
        call(&mut k, "XcSHAInit", &[context]);
        call(&mut k, "XcSHAUpdate", &[context, input, 1]);
        call(&mut k, "XcSHAUpdate", &[context, input + 1, 2]);
        call(&mut k, "XcSHAFinal", &[context, digest]);
        assert_eq!(
            hex(&k.read(digest, 20).unwrap()),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
    }

    #[test]
    fn hmac_matches_rfc_2202_over_two_pieces() {
        let mut k = kernel();
        let (key, data, digest) = (SCRATCH, SCRATCH + 0x100, SCRATCH + 0x200);
        k.write(key, b"Jefe").unwrap();
        k.write(data, b"what do ya want for nothing?").unwrap();
        call(&mut k, "XcHMAC", &[key, 4, data, 10, data + 10, 18, digest]);
        assert_eq!(
            hex(&k.read(digest, 20).unwrap()),
            "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79"
        );
    }

    #[test]
    fn rc4_matches_the_known_vector_and_decrypts_back() {
        let mut k = kernel();
        let (state, key, data) = (SCRATCH, SCRATCH + 0x200, SCRATCH + 0x300);
        k.write(key, b"Key").unwrap();
        k.write(data, b"Plaintext").unwrap();
        call(&mut k, "XcRC4Key", &[state, 3, key]);
        call(&mut k, "XcRC4Crypt", &[state, 9, data]);
        assert_eq!(hex(&k.read(data, 9).unwrap()), "bbf316e8d940af0ad3");
        call(&mut k, "XcRC4Key", &[state, 3, key]);
        call(&mut k, "XcRC4Crypt", &[state, 9, data]);
        assert_eq!(k.read(data, 9).unwrap(), b"Plaintext");
    }

    #[test]
    fn des_parity_block_and_cbc() {
        let mut k = kernel();
        let (key, table, input, output, feedback) = (
            SCRATCH,
            SCRATCH + 0x40,
            SCRATCH + 0x80,
            SCRATCH + 0xC0,
            SCRATCH + 0x100,
        );
        k.write(key, &[0x00, 0x01, 0x02, 0x03, 0xFE, 0xFF, 0x80, 0x7F])
            .unwrap();
        call(&mut k, "XcDESKeyParity", &[key, 8]);
        assert_eq!(
            k.read(key, 8).unwrap(),
            [0x01, 0x01, 0x02, 0x02, 0xFE, 0xFE, 0x80, 0x7F]
        );
        // The classic DES vector: key 133457799BBCDFF1, "0123456789ABCDEF" to 85E813540F0AB405.
        k.write(key, &[0x13, 0x34, 0x57, 0x79, 0x9B, 0xBC, 0xDF, 0xF1])
            .unwrap();
        call(&mut k, "XcKeyTable", &[0, table, key]);
        k.write(input, &[0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF])
            .unwrap();
        call(&mut k, "XcBlockCrypt", &[0, output, input, table, 1]);
        assert_eq!(hex(&k.read(output, 8).unwrap()), "85e813540f0ab405");
        call(&mut k, "XcBlockCrypt", &[0, output, output, table, 0]);
        assert_eq!(hex(&k.read(output, 8).unwrap()), "0123456789abcdef");
        // CBC with a zero feedback block: the first block is the plain one's, and decryption
        // with the same starting feedback gives the input back.
        k.write(input + 8, b"second!!").unwrap();
        k.write(feedback, &[0; 8]).unwrap();
        call(
            &mut k,
            "XcBlockCryptCBC",
            &[0, 16, output, input, table, 1, feedback],
        );
        assert_eq!(hex(&k.read(output, 8).unwrap()), "85e813540f0ab405");
        assert_eq!(k.read(feedback, 8).unwrap(), k.read(output + 8, 8).unwrap());
        k.write(feedback, &[0; 8]).unwrap();
        call(
            &mut k,
            "XcBlockCryptCBC",
            &[0, 16, output, output, table, 0, feedback],
        );
        assert_eq!(k.read(output, 16).unwrap(), k.read(input, 16).unwrap());
    }

    #[test]
    fn mod_exp_and_a_small_rsa_key() {
        let mut k = kernel();
        let (out, base, exponent, modulus) = (SCRATCH, SCRATCH + 8, SCRATCH + 16, SCRATCH + 24);
        k.w64(base, 4).unwrap();
        k.w64(exponent, 13).unwrap();
        k.w64(modulus, 497).unwrap();
        let args = [out, base, exponent, modulus, 2];
        assert_eq!(call(&mut k, "XcModExp", &args), 1);
        assert_eq!(k.r64(out), Ok(445));
        k.w64(modulus, 0).unwrap();
        assert_eq!(call(&mut k, "XcModExp", &args), 0);
        // n = 61 * 53 = 3233, e = 17: 65 encrypts to 2790.
        let key = SCRATCH + 0x100;
        k.write(key, b"RSA1").unwrap();
        for (i, field) in [4u32, 12, 3, 17].into_iter().enumerate() {
            k.w32(key + 4 + 4 * i as u32, field).unwrap();
        }
        k.w32(key + KEY_HEADER, 3233).unwrap();
        let (input, output) = (SCRATCH + 0x200, SCRATCH + 0x210);
        k.w32(input, 65).unwrap();
        assert_eq!(call(&mut k, "XcPKGetKeyLen", &[key]), 4);
        assert_eq!(call(&mut k, "XcPKEncPublic", &[key, input, output]), 1);
        assert_eq!(k.r32(output), Ok(2790));
        assert_eq!(
            call(&mut k, "XcPKDecPrivate", &[key, input, output]),
            0,
            "a public key is not a private one"
        );
    }

    const KEY: u32 = SCRATCH + 0x100;
    const SIGNATURE: u32 = SCRATCH + 0x400;
    const DIGEST: u32 = SCRATCH + 0x600;
    /// A 2048-bit key: 256 bytes of modulus in a 264-byte buffer, 255 bytes of data.
    const KEY_LEN: usize = 264;
    const DATA_LEN: usize = 255;

    /// The block `XCApplyPKCS1SigningFmt` builds, least significant byte first: the digest
    /// reversed, the `DigestInfo` `prefix`, a zero, 0xFF padding, a 1 and a zero.
    fn signing_block(digest: &[u8; 20], prefix: &[u8]) -> Vec<u8> {
        let mut block = vec![0xFF; KEY_LEN];
        block[..20].copy_from_slice(&digest.iter().rev().copied().collect::<Vec<_>>());
        block[20..20 + prefix.len()].copy_from_slice(prefix);
        block[20 + prefix.len()] = 0;
        block[DATA_LEN - 1] = 1;
        block[DATA_LEN] = 0;
        block[DATA_LEN + 1..].fill(0);
        block
    }

    /// Runs the export on `block` taken as the signature, with the public exponent 1 so that
    /// opening it returns it unchanged; returns what the call returned.
    fn verify(block: &[u8], digest: &[u8; 20]) -> u32 {
        let mut key = Vec::from(*b"RSA1");
        for field in [KEY_LEN as u32, 2048, DATA_LEN as u32, 1] {
            key.extend_from_slice(&field.to_le_bytes());
        }
        // A modulus above every block: its top byte is 0x80, the blocks' is zero.
        let mut modulus = vec![0xFF; KEY_LEN];
        modulus[DATA_LEN] = 0x80;
        modulus[DATA_LEN + 1..].fill(0);
        key.extend_from_slice(&modulus);
        let mut k = kernel();
        k.write(KEY, &key).unwrap();
        k.write(SIGNATURE, block).unwrap();
        k.write(DIGEST, digest).unwrap();
        call(&mut k, "XcVerifyPKCS1Signature", &[SIGNATURE, KEY, DIGEST])
    }

    fn digest() -> [u8; 20] {
        core::array::from_fn(|i| (i as u8) * 7 + 1)
    }

    #[test]
    fn pkcs1_accepts_the_kernels_block_with_either_digest_info_or_none() {
        for prefix in [SHA_ENCODINGS[0], SHA_ENCODINGS[1], &[]] {
            assert_eq!(verify(&signing_block(&digest(), prefix), &digest()), 1);
        }
    }

    #[test]
    fn pkcs1_wants_the_digest_reversed_at_the_start() {
        // The digest in its own order, as a big-endian block would end, is not accepted.
        let mut block = signing_block(&digest(), SHA_ENCODINGS[0]);
        block[..20].copy_from_slice(&digest());
        assert_eq!(verify(&block, &digest()), 0);
        let mut other = digest();
        other[0] ^= 1;
        assert_eq!(
            verify(&signing_block(&digest(), SHA_ENCODINGS[0]), &other),
            0
        );
    }

    #[test]
    fn pkcs1_checks_the_padding_and_the_signature_range() {
        let good = signing_block(&digest(), SHA_ENCODINGS[0]);
        // A hole in the 0xFF run, a wrong block type, a wrong separator, a wrong prefix.
        for broken in [60, DATA_LEN - 1, DATA_LEN, 25] {
            let mut block = good.clone();
            block[broken] ^= 0x55;
            assert_eq!(verify(&block, &digest()), 0, "byte {broken}");
        }
        // A signature at or past the modulus.
        let mut huge = good;
        huge[DATA_LEN] = 0xFF;
        assert_eq!(verify(&huge, &digest()), 0);
    }

    #[test]
    fn pkcs1_block_check_matches_the_kernels_loop_bounds() {
        // No room for padding at all: the zero separator sits right before the 1.
        let mut block = vec![0u8; 40];
        block[..20].copy_from_slice(&digest().iter().rev().copied().collect::<Vec<_>>());
        block[21] = 1;
        assert!(pkcs1_block_matches(&block, 22, &digest()));
        assert!(!pkcs1_block_matches(&block, 21, &digest()));
        assert!(!pkcs1_block_matches(&block[..22], 22, &digest()));
    }
}
