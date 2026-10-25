//! Serde for the arrays longer than the 32 elements serde supports itself.

/// A boxed array, as a sequence.
pub mod boxed_array {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// # Errors
    ///
    /// The serializer's.
    #[allow(clippy::borrowed_box)] // serde's `with` passes the field as it is.
    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        value: &Box<[T; N]>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_slice().serialize(serializer)
    }

    /// # Errors
    ///
    /// The deserializer's, or a sequence of the wrong length.
    pub fn deserialize<'de, D: Deserializer<'de>, T: Deserialize<'de>, const N: usize>(
        deserializer: D,
    ) -> Result<Box<[T; N]>, D::Error> {
        let items = Vec::<T>::deserialize(deserializer)?;
        let len = items.len();
        items
            .into_boxed_slice()
            .try_into()
            .map_err(|_| D::Error::invalid_length(len, &"the array's length"))
    }
}

/// A map of 512-byte sectors, as a sequence of (number, bytes).
pub mod sectors {
    use std::collections::HashMap;

    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    /// # Errors
    ///
    /// The serializer's.
    #[allow(clippy::implicit_hasher)] // `with` names this one map type.
    pub fn serialize<S: Serializer>(
        map: &HashMap<u64, Box<[u8; 512]>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(map.iter().map(|(number, bytes)| (number, bytes.as_slice())))
    }

    /// # Errors
    ///
    /// The deserializer's, or a sector that is not 512 bytes.
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<HashMap<u64, Box<[u8; 512]>>, D::Error> {
        Vec::<(u64, Vec<u8>)>::deserialize(deserializer)?
            .into_iter()
            .map(|(number, bytes)| {
                let len = bytes.len();
                bytes
                    .into_boxed_slice()
                    .try_into()
                    .map(|sector| (number, sector))
                    .map_err(|_| D::Error::invalid_length(len, &"512 bytes"))
            })
            .collect()
    }
}
