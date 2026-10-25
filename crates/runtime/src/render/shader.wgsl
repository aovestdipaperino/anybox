// The renderer's shader (spec 11.4, 11.5). The fragment stage mirrors `combiner.rs`, which is its
// reference and is unit tested; keep the two in step.

struct Params {
    // x: combiner in use, y: stage count, z: control, w: final word 1.
    // x: final word 0, y: alpha test function, z: alpha reference, w: alpha kill bits (a stage
    // each).
    // The four stages' mip biases (f32 bits).
    header: array<vec4<u32>, 3>,
    // Colour inputs, colour outputs, alpha inputs, alpha outputs: two vec4s (8 words) each.
    words: array<vec4<u32>, 8>,
    // Constant 0 (8), constant 1 (8), final constants (2).
    constants: array<vec4<f32>, 18>,
    // The texture stages (`stages.rs` is their reference): x each stage's mode (`Mode`) in a
    // byte, bit 5 for a 3D texture and 6 for a cube map; y the dot mappings; z the input stages
    // of stages 2 and 3, four bits each.
    stages: vec4<u32>,
    // Each stage's coordinate scales, two a stage.
    scales: array<vec4<f32>, 2>,
    // Each stage's bump matrix, m00, m01, m10, m11.
    bump: array<vec4<f32>, 4>,
    // Each stage's bump luminance scale, then each one's offset.
    luminance: array<vec4<f32>, 2>,
    // The pixel state: the four stages' colour keys; the texel bits each compares; x the colour
    // key modes (two bits a stage: 1 clears a matching texel's alpha, 2 the whole texel, 3 drops
    // the fragment), y the window clip (0 none, 1 inside a rectangle, 2 outside them all), z bit
    // 0 drops depth outside the range and bit 1 w-buffers, w the largest depth (f32 bits); x, y
    // the depth range (f32 bits).
    pixel: array<vec4<u32>, 4>,
    // The window clip's rectangles: first pixel across and down, then the pixel after the last.
    window: array<vec4<u32>, 8>,
}

@group(1) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var texture0: texture_2d<f32>;
@group(0) @binding(2) var sampler0: sampler;
@group(0) @binding(3) var texture1: texture_2d<f32>;
@group(0) @binding(4) var sampler1: sampler;
@group(0) @binding(5) var texture2: texture_2d<f32>;
@group(0) @binding(6) var sampler2: sampler;
@group(0) @binding(7) var texture3: texture_2d<f32>;
@group(0) @binding(8) var sampler3: sampler;
// Each stage's texture again where it is 3D or a cube map (white otherwise).
@group(0) @binding(9) var volume0: texture_3d<f32>;
@group(0) @binding(10) var volume1: texture_3d<f32>;
@group(0) @binding(11) var volume2: texture_3d<f32>;
@group(0) @binding(12) var volume3: texture_3d<f32>;
@group(0) @binding(13) var cube0: texture_cube<f32>;
@group(0) @binding(14) var cube1: texture_cube<f32>;
@group(0) @binding(15) var cube2: texture_cube<f32>;
@group(0) @binding(16) var cube3: texture_cube<f32>;

struct VertexIn {
    @location(0) clip: vec4<f32>,
    @location(1) colour: vec4<f32>,
    @location(2) fog: vec4<f32>,
    @location(3) t0: vec4<f32>,
    @location(4) t1: vec4<f32>,
    @location(5) t2: vec4<f32>,
    @location(6) t3: vec4<f32>,
    @location(7) specular: vec4<f32>,
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec4<f32>,
    @location(1) fog: vec4<f32>,
    @location(2) t0: vec4<f32>,
    @location(3) t1: vec4<f32>,
    @location(4) t2: vec4<f32>,
    @location(5) t3: vec4<f32>,
    @location(6) specular: vec4<f32>,
    // Clip-space z and w: their quotient is the depth in the viewport's 0..1 before it is
    // clamped, and w the eye's.
    @location(7) clip_zw: vec2<f32>,
}

@vertex
fn vs(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = v.clip;
    out.colour = v.colour;
    out.fog = v.fog;
    out.t0 = v.t0;
    out.t1 = v.t1;
    out.t2 = v.t2;
    out.t3 = v.t3;
    out.specular = v.specular;
    out.clip_zw = v.clip.zw;
    return out;
}

// Word `i` of array `a` (0 colour inputs, 1 colour outputs, 2 alpha inputs, 3 alpha outputs).
fn word(a: u32, i: u32) -> u32 {
    return params.words[a * 2u + i / 4u][i % 4u];
}

fn argb(index: u32) -> vec4<f32> {
    return params.constants[index];
}

fn map(mapping: u32, x: f32) -> f32 {
    let positive = max(x, 0.0);
    switch mapping {
        case 0u: { return positive; }
        case 1u: { return 1.0 - clamp(x, 0.0, 1.0); }
        case 2u: { return 2.0 * positive - 1.0; }
        case 3u: { return 1.0 - 2.0 * positive; }
        case 4u: { return positive - 0.5; }
        case 5u: { return 0.5 - positive; }
        case 6u: { return x; }
        default: { return -x; }
    }
}

fn map3(mapping: u32, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(map(mapping, v.x), map(mapping, v.y), map(mapping, v.z));
}

fn scale(w: u32, x: f32) -> f32 {
    var y = x;
    switch (w >> 15u) & 7u {
        case 1u: { y = x - 0.5; }
        case 2u: { y = x * 2.0; }
        case 3u: { y = (x - 0.5) * 2.0; }
        case 4u: { y = x * 4.0; }
        case 6u: { y = x * 0.5; }
        default: {}
    }
    return clamp(y, -1.0, 1.0);
}

fn scale3(w: u32, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(scale(w, v.x), scale(w, v.y), scale(w, v.z));
}

var<private> regs: array<vec4<f32>, 16>;

fn colour_input(byte: u32) -> vec3<f32> {
    let r = regs[byte & 15u];
    var v = r.xyz;
    if (byte & 16u) != 0u { v = vec3<f32>(r.w); }
    return map3((byte >> 5u) & 7u, v);
}

fn alpha_input(byte: u32) -> f32 {
    let r = regs[byte & 15u];
    var v = r.z;
    if (byte & 16u) != 0u { v = r.w; }
    return map((byte >> 5u) & 7u, v);
}

fn final_input(byte: u32) -> vec3<f32> {
    let r = regs[byte & 15u];
    var v = r.xyz;
    if (byte & 16u) != 0u { v = vec3<f32>(r.w); }
    v = clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
    if (byte & 32u) != 0u { v = vec3<f32>(1.0) - v; }
    return v;
}

fn product(x: vec3<f32>, y: vec3<f32>, is_dot: bool) -> vec3<f32> {
    if is_dot { return vec3<f32>(dot(x, y)); }
    return x * y;
}

fn general_stage(i: u32) {
    let control = params.header[0].z;
    var c0 = 0u;
    var c1 = 0u;
    if (control & 16u) != 0u { c0 = i; }
    if (control & 256u) != 0u { c1 = i; }
    regs[1] = argb(c0);
    regs[2] = argb(8u + c1);
    var mux = false;
    let spare_alpha = regs[12].w;
    if (control & 1u) != 0u {
        mux = spare_alpha >= 0.5;
    } else {
        mux = (i32(round(spare_alpha * 255.0)) & 1) != 0;
    }

    let out = word(1u, i);
    let inputs = word(0u, i);
    let a = colour_input(inputs >> 24u);
    let b = colour_input((inputs >> 16u) & 255u);
    let c = colour_input((inputs >> 8u) & 255u);
    let d = colour_input(inputs & 255u);
    var ab = product(a, b, (out & (1u << 13u)) != 0u);
    var cd = product(c, d, (out & (1u << 12u)) != 0u);
    var sum = ab + cd;
    if (out & (1u << 14u)) != 0u {
        sum = select(ab, cd, mux);
    }
    ab = scale3(out, ab);
    cd = scale3(out, cd);
    sum = scale3(out, sum);

    let alpha_out = word(3u, i);
    let alpha_inputs = word(2u, i);
    var alpha_ab = alpha_input(alpha_inputs >> 24u) * alpha_input((alpha_inputs >> 16u) & 255u);
    var alpha_cd = alpha_input((alpha_inputs >> 8u) & 255u) * alpha_input(alpha_inputs & 255u);
    var alpha_sum = alpha_ab + alpha_cd;
    if (alpha_out & (1u << 14u)) != 0u {
        alpha_sum = select(alpha_ab, alpha_cd, mux);
    }
    alpha_ab = scale(alpha_out, alpha_ab);
    alpha_cd = scale(alpha_out, alpha_cd);
    alpha_sum = scale(alpha_out, alpha_sum);

    let cd_dst = out & 15u;
    let ab_dst = (out >> 4u) & 15u;
    let sum_dst = (out >> 8u) & 15u;
    if cd_dst != 0u { regs[cd_dst] = vec4<f32>(cd, regs[cd_dst].w); }
    if ab_dst != 0u { regs[ab_dst] = vec4<f32>(ab, regs[ab_dst].w); }
    if sum_dst != 0u { regs[sum_dst] = vec4<f32>(sum, regs[sum_dst].w); }
    if (out & (1u << 18u)) != 0u && cd_dst != 0u { regs[cd_dst].w = cd.z; }
    if (out & (1u << 19u)) != 0u && ab_dst != 0u { regs[ab_dst].w = ab.z; }
    let acd = alpha_out & 15u;
    let aab = (alpha_out >> 4u) & 15u;
    let asum = (alpha_out >> 8u) & 15u;
    if acd != 0u { regs[acd].w = alpha_cd; }
    if aab != 0u { regs[aab].w = alpha_ab; }
    if asum != 0u { regs[asum].w = alpha_sum; }
}

fn complement(x: vec3<f32>, on: bool) -> vec3<f32> {
    if on { return vec3<f32>(1.0) - clamp(x, vec3<f32>(0.0), vec3<f32>(1.0)); }
    return x;
}

fn combine(t0: vec4<f32>, t1: vec4<f32>, t2: vec4<f32>, t3: vec4<f32>, colour: vec4<f32>,
           fog: vec4<f32>, specular: vec4<f32>) -> vec4<f32> {
    for (var r = 0u; r < 16u; r = r + 1u) { regs[r] = vec4<f32>(0.0); }
    regs[3] = vec4<f32>(fog.xyz, 1.0 - fog.w);
    regs[4] = colour;
    regs[5] = specular;
    regs[8] = t0;
    regs[9] = t1;
    regs[10] = t2;
    regs[11] = t3;
    regs[12] = vec4<f32>(0.0, 0.0, 0.0, t0.w);
    let stages = params.header[0].y;
    for (var i = 0u; i < stages; i = i + 1u) { general_stage(i); }
    regs[1] = argb(16u);
    regs[2] = argb(17u);
    let final1 = params.header[0].w;
    let flags = final1 & 255u;
    regs[14] = vec4<f32>(complement(regs[5].xyz, (flags & 64u) != 0u) + complement(regs[12].xyz, (flags & 32u) != 0u), 0.0);
    let e = final_input(final1 >> 24u);
    let f = final_input((final1 >> 16u) & 255u);
    regs[15] = vec4<f32>(e * f, 0.0);
    let final0 = params.header[1].x;
    let a = final_input(final0 >> 24u);
    let b = final_input((final0 >> 16u) & 255u);
    let c = final_input((final0 >> 8u) & 255u);
    let d = final_input(final0 & 255u);
    let alpha = final_input(((final1 >> 8u) & 255u) | 16u).x;
    let rgb = clamp(a * b + (vec3<f32>(1.0) - a) * c + d, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb, clamp(alpha, 0.0, 1.0));
}

fn compare(function: u32, value: u32, reference: u32) -> bool {
    switch function & 15u {
        case 0u: { return false; }
        case 1u: { return value < reference; }
        case 2u: { return value == reference; }
        case 3u: { return value <= reference; }
        case 4u: { return value > reference; }
        case 5u: { return value != reference; }
        case 6u: { return value >= reference; }
        default: { return true; }
    }
}

// The texture stages, mirroring `stages.rs`: each stage's coordinates `s, t, r, q`, what the
// stages before gave, and their dot products.
var<private> coords: array<vec4<f32>, 4>;
var<private> texels: array<vec4<f32>, 4>;
var<private> dots: array<f32, 4>;

fn mode_of(stage: u32) -> u32 {
    return (params.stages.x >> (8u * stage)) & 31u;
}

// 0 flat, 1 a 3D texture, 2 a cube map.
fn shape_of(stage: u32) -> u32 {
    return (params.stages.x >> (8u * stage + 5u)) & 3u;
}

fn input_of(stage: u32) -> u32 {
    if stage == 2u { return params.stages.z & 15u; }
    if stage == 3u { return (params.stages.z >> 4u) & 15u; }
    return 0u;
}

fn scale_of(stage: u32) -> vec2<f32> {
    let v = params.scales[stage / 2u];
    if (stage & 1u) != 0u { return v.zw; }
    return v.xy;
}

fn bias_of(stage: u32) -> f32 {
    return bitcast<f32>(params.header[2][stage]);
}

// Whether the mode reads the stage's texture.
fn samples(mode: u32) -> bool {
    return !(mode == 1u || mode == 2u || mode == 3u || mode == 8u || mode == 10u);
}

fn sign3(x: f32) -> f32 {
    let v = x * 255.0;
    if v >= 128.0 { return (v - 256.0) / 127.0; }
    return v / 127.0;
}

fn dot_mapped(c: vec4<f32>, stage: u32) -> vec3<f32> {
    if stage == 0u { return c.xyz; }
    switch (params.stages.y >> (4u * (stage - 1u))) & 15u {
        case 1u: { return (c.xyz * 255.0 - 128.0) / 127.0; }
        case 2u: {
            let v = c.xyz * 255.0;
            return select((v + 0.5) / 127.5, (v - 255.5) / 127.5, v >= vec3<f32>(128.0));
        }
        case 3u: { return vec3<f32>(sign3(c.x), sign3(c.y), sign3(c.z)); }
        case 4u: {
            let b = vec4<u32>(round(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
            return vec3<f32>(f32((b.w << 8u) | b.x) / 65535.0, f32((b.y << 8u) | b.z) / 65535.0, 1.0);
        }
        default: { return c.xyz; }
    }
}

fn cube_to_flat(c: vec3<f32>) -> vec2<f32> {
    let a = abs(c);
    if a.x > a.y && a.x > a.z {
        if c.x > 0.0 { return vec2<f32>(-c.z, c.y) / a.x; }
        return vec2<f32>(c.z, c.y) / a.x;
    }
    if a.y > a.x && a.y > a.z {
        if c.y > 0.0 { return vec2<f32>(c.x, -c.z) / a.y; }
        return vec2<f32>(c.x, c.z) / a.y;
    }
    if c.z > 0.0 { return vec2<f32>(c.x, c.y) / a.z; }
    return vec2<f32>(-c.x, c.y) / a.z;
}

// A direction as the cube lookup takes it, or its place on the face it points to for a texture
// that is not a cube map.
fn toward(stage: u32, c: vec3<f32>) -> vec3<f32> {
    if shape_of(stage) == 2u { return c; }
    return vec3<f32>(cube_to_flat(c), 0.0);
}

fn dot_of(stage: u32) -> f32 {
    return dot(coords[stage].xyz, dot_mapped(texels[input_of(stage)], stage));
}

fn coordinates(stage: u32) -> vec3<f32> {
    let p = coords[stage];
    let source = texels[input_of(stage)];
    switch mode_of(stage) {
        case 0u: {
            let st = p.xy / p.w;
            if shape_of(stage) == 2u { return normalize(vec3<f32>(1.0, st.y, -st.x)); }
            return vec3<f32>(st, 0.0);
        }
        case 4u: { return p.xyz / p.w; }
        case 5u: { return toward(stage, p.xyz); }
        case 6u, 7u: {
            let m = params.bump[stage];
            let du = sign3(source.z);
            let dv = sign3(source.y);
            return vec3<f32>(p.x + m.x * du + m.z * dv, p.y + m.y * du + m.w * dv, p.z);
        }
        case 9u: { return vec3<f32>(dots[stage - 1u], dots[stage], 0.0); }
        case 11u: {
            let n = vec3<f32>(dots[1], dots[2], dot(coords[3].xyz, dot_mapped(texels[input_of(3u)], 3u)));
            return toward(stage, n);
        }
        case 12u: {
            let n = vec3<f32>(dots[1], dots[2], dots[3]);
            let e = vec3<f32>(coords[1].w, coords[2].w, coords[3].w);
            return toward(stage, 2.0 * n * dot(n, e) / dot(n, n) - e);
        }
        case 13u: { return vec3<f32>(dots[1], dots[2], dots[3]); }
        case 14u: { return toward(stage, vec3<f32>(dots[1], dots[2], dots[3])); }
        case 15u: { return vec3<f32>(source.w, source.x, 0.0); }
        case 16u: { return vec3<f32>(source.y, source.z, 0.0); }
        default: { return p.xyz; }
    }
}

// Which lookup the stage takes: 0 flat, 1 the 3D texture, 2 the cube map.
fn lookup_of(stage: u32) -> u32 {
    let mode = mode_of(stage);
    let shape = shape_of(stage);
    if shape == 2u && (mode == 0u || mode == 5u || mode == 11u || mode == 12u || mode == 14u) {
        return 2u;
    }
    if shape == 1u && (mode == 0u || mode == 4u || mode == 6u || mode == 7u || mode == 13u) {
        return 1u;
    }
    return 0u;
}

// What the stage gives the combiners from the texel its lookup found.
fn staged(stage: u32, texel: vec4<f32>) -> vec4<f32> {
    switch mode_of(stage) {
        case 1u: { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
        case 2u: { return clamp(coords[stage], vec4<f32>(0.0), vec4<f32>(1.0)); }
        case 3u, 8u, 10u: { return vec4<f32>(0.0); }
        case 7u: {
            let red = texels[input_of(stage)].x;
            return texel * (params.luminance[0][stage] * red + params.luminance[1][stage]);
        }
        default: { return texel; }
    }
}

// Whether the stage's alpha kill drops the fragment.
fn killed(stage: u32) -> bool {
    return samples(mode_of(stage)) && ((params.header[1].w >> stage) & 1u) != 0u
        && texels[stage].w == 0.0;
}

// A stage's colour key mode.
fn key_mode(stage: u32) -> u32 {
    return (params.pixel[2].x >> (2u * stage)) & 3u;
}

// Whether a texel matches its stage's colour key, compared as 0xAARRGGBB through the mask.
fn keyed(stage: u32, texel: vec4<f32>) -> bool {
    let c = vec4<u32>(clamp(texel, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + 0.5);
    let argb = (c.w << 24u) | (c.x << 16u) | (c.y << 8u) | c.z;
    let mask = params.pixel[1][stage];
    return (argb & mask) == (params.pixel[0][stage] & mask);
}

// The colour key of `stage` applied to its texel; mode 3 drops the fragment.
fn key(stage: u32, texel: vec4<f32>) -> vec4<f32> {
    let mode = key_mode(stage);
    if mode == 0u || !keyed(stage, texel) { return texel; }
    if mode == 3u { discard; }
    if mode == 2u { return vec4<f32>(0.0); }
    return vec4<f32>(texel.xyz, 0.0);
}

// Depth outside the range drops the fragment unless it is clamped; a hair outside is kept, as
// the hardware compares in its own units.
@fragment
fn fs(v: VertexOut) -> @location(0) vec4<f32> {
    let depth = v.clip_zw.x / v.clip_zw.y;
    if (params.pixel[2].z & 1u) != 0u && (depth < -1e-6 || depth > 1.0 + 1e-6) {
        discard;
    }
    return shade(v);
}

struct DepthOut {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

// W-buffering: the depth is the interpolated w over the buffer's largest depth, a w behind the
// eye the farthest.
@fragment
fn fs_w_buffer(v: VertexOut) -> DepthOut {
    var z = v.clip_zw.y / bitcast<f32>(params.pixel[2].w);
    if !(z > 0.0) { z = 3.4e38; }
    let near = bitcast<f32>(params.pixel[3].x);
    let far = bitcast<f32>(params.pixel[3].y);
    if (params.pixel[2].z & 1u) != 0u && (z < near || z > far) { discard; }
    var out: DepthOut;
    out.colour = shade(v);
    out.depth = clamp(clamp(z, near, far), 0.0, 1.0);
    return out;
}

fn shade(v: VertexOut) -> vec4<f32> {
    coords = array<vec4<f32>, 4>(v.t0, v.t1, v.t2, v.t3);
    // Each stage takes the one lookup its mode and texture need, none when its mode reads no
    // texture: both are the batch's, so the branches are uniform.
    var s = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var c = coordinates(0u);
    if samples(mode_of(0u)) {
        switch lookup_of(0u) {
            case 1u: { s = textureSampleBias(volume0, sampler0, c, bias_of(0u)); }
            case 2u: { s = textureSampleBias(cube0, sampler0, c, bias_of(0u)); }
            default: { s = textureSampleBias(texture0, sampler0, c.xy * scale_of(0u), bias_of(0u)); }
        }
    }
    texels[0] = staged(0u, key(0u, s));

    dots[1] = dot_of(1u);
    c = coordinates(1u);
    s = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if samples(mode_of(1u)) {
        switch lookup_of(1u) {
            case 1u: { s = textureSampleBias(volume1, sampler1, c, bias_of(1u)); }
            case 2u: { s = textureSampleBias(cube1, sampler1, c, bias_of(1u)); }
            default: { s = textureSampleBias(texture1, sampler1, c.xy * scale_of(1u), bias_of(1u)); }
        }
    }
    texels[1] = staged(1u, key(1u, s));

    dots[2] = dot_of(2u);
    c = coordinates(2u);
    s = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if samples(mode_of(2u)) {
        switch lookup_of(2u) {
            case 1u: { s = textureSampleBias(volume2, sampler2, c, bias_of(2u)); }
            case 2u: { s = textureSampleBias(cube2, sampler2, c, bias_of(2u)); }
            default: { s = textureSampleBias(texture2, sampler2, c.xy * scale_of(2u), bias_of(2u)); }
        }
    }
    texels[2] = staged(2u, key(2u, s));

    dots[3] = dot_of(3u);
    c = coordinates(3u);
    s = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if samples(mode_of(3u)) {
        switch lookup_of(3u) {
            case 1u: { s = textureSampleBias(volume3, sampler3, c, bias_of(3u)); }
            case 2u: { s = textureSampleBias(cube3, sampler3, c, bias_of(3u)); }
            default: { s = textureSampleBias(texture3, sampler3, c.xy * scale_of(3u), bias_of(3u)); }
        }
    }
    texels[3] = staged(3u, key(3u, s));

    if killed(0u) || killed(1u) || killed(2u) || killed(3u) {
        discard;
    }
    let t0 = texels[0];
    let t1 = texels[1];
    var out: vec4<f32>;
    if params.header[0].x != 0u {
        out = combine(t0, t1, texels[2], texels[3], v.colour, v.fog, v.specular);
    } else {
        // Without a combiner program, a stage that reads no texture leaves the colour as it is.
        let white = vec4<f32>(1.0);
        let lit = select(white, t0, samples(mode_of(0u))) * select(white, t1, samples(mode_of(1u)))
            * v.colour;
        out = vec4<f32>(mix(lit.xyz, v.fog.xyz, v.fog.w), lit.w);
    }
    let function = params.header[1].y;
    let alpha = u32(round(clamp(out.w, 0.0, 1.0) * 255.0));
    if function >= 0x200u && function < 0x207u && !compare(function - 0x200u, alpha, params.header[1].z) {
        discard;
    }
    // The window clip: inside one of the rectangles (1), or outside them all (2).
    let window = params.pixel[2].y;
    if window != 0u {
        let pixel = vec2<u32>(v.position.xy);
        var inside = false;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let r = params.window[i];
            if all(pixel >= r.xy) && all(pixel < r.zw) { inside = true; }
        }
        if inside == (window == 2u) { discard; }
    }
    return out;
}
