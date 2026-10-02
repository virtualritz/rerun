// Fidget tape evaluator, vendored for akatela SPEC-109.
//
// Adapted from `fidget-wgpu`'s `shaders/{common,float_ops,stack,tape_interpreter}.wgsl`
// (fidget 0.5.1, git `dad9e3f`). The opcode table and bytecode format are
// fidget's; the pass that drives this is re_renderer's (SPEC-109 R1, T011).
// Only the tape and variable buffers move from a packed storage-buffer
// `Config` into two read-only bindings. Keep the opcode constants in sync
// with `fidget_bytecode::iter_ops()`; `tests/sdf_tape.rs` is the drift guard.

// --- Opcodes (fidget_bytecode::BytecodeOp) ---
const OP_OUTPUT: u32 = 0u;
const OP_INPUT: u32 = 1u;
const OP_COPY: u32 = 2u;
const OP_NEG: u32 = 3u;
const OP_ABS: u32 = 4u;
const OP_RECIP: u32 = 5u;
const OP_SQRT: u32 = 6u;
const OP_SQUARE: u32 = 7u;
const OP_FLOOR: u32 = 8u;
const OP_CEIL: u32 = 9u;
const OP_ROUND: u32 = 10u;
const OP_NOT: u32 = 11u;
const OP_RAND: u32 = 12u;
const OP_SIN: u32 = 13u;
const OP_COS: u32 = 14u;
const OP_TAN: u32 = 15u;
const OP_ASIN: u32 = 16u;
const OP_ACOS: u32 = 17u;
const OP_ATAN: u32 = 18u;
const OP_EXP: u32 = 19u;
const OP_LN: u32 = 20u;
const OP_ADD: u32 = 21u;
const OP_SUB: u32 = 22u;
const OP_MUL: u32 = 23u;
const OP_DIV: u32 = 24u;
const OP_ATAN2: u32 = 25u;
const OP_COMPARE: u32 = 26u;
const OP_MIX: u32 = 27u;
const OP_MOD: u32 = 28u;
const OP_MIN: u32 = 29u;
const OP_MAX: u32 = 30u;
const OP_AND: u32 = 31u;
const OP_OR: u32 = 32u;
const OP_MEM: u32 = 33u;
const OP_JUMP: u32 = 0xFFu;

/// Maximum registers a bytecode tape may use (register 255 is reserved).
const REG_COUNT: u32 = 255u;

// --- Tape and variables (group 1 bindings 1 and 2) ---
struct TapeWord {
    op: u32,
    imm: u32,
}

@group(1) @binding(1)
var<storage, read> tape_data: array<TapeWord>;

@group(1) @binding(2)
var<storage, read> var_values: array<f32>;

// --- common.wgsl ---
fn nan_f32() -> f32 {
    // Workaround for https://github.com/gpuweb/gpuweb/issues/3749
    let bits = 0xffffffffu;
    return bitcast<f32>(bits);
}

fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if r < 0.0 {
        return r + abs(rhs);
    } else {
        return r;
    }
}

fn hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

fn rand(seed: u32) -> f32 {
    let h = hash(seed);
    let bits = (h >> 9) | 0x3f800000;
    return bitcast<f32>(bits) - 1.0;
}

fn mix_u32(a: u32, b: u32) -> u32 {
    return hash(a + hash(b));
}

// --- float_ops.wgsl ---
struct Value {
    v: f32,
}

fn build_imm(imm: f32) -> Value {
    return Value(imm);
}

fn op_abs(lhs: Value) -> Value { return Value(abs(lhs.v)); }
fn op_acos(lhs: Value) -> Value { return Value(acos(lhs.v)); }
fn op_cos(lhs: Value) -> Value { return Value(cos(lhs.v)); }
fn op_asin(lhs: Value) -> Value { return Value(asin(lhs.v)); }
fn op_atan(lhs: Value) -> Value { return Value(atan(lhs.v)); }
fn op_ceil(lhs: Value) -> Value { return Value(ceil(lhs.v)); }
fn op_floor(lhs: Value) -> Value { return Value(floor(lhs.v)); }
fn op_log(lhs: Value) -> Value { return Value(log(lhs.v)); }
fn op_recip(lhs: Value) -> Value { return Value(1.0 / lhs.v); }
fn op_round(lhs: Value) -> Value { return Value(round(lhs.v)); }
fn op_sin(lhs: Value) -> Value { return Value(sin(lhs.v)); }
fn op_tan(lhs: Value) -> Value { return Value(tan(lhs.v)); }
fn op_exp(lhs: Value) -> Value { return Value(exp(lhs.v)); }
fn op_add(lhs: Value, rhs: Value) -> Value { return Value(lhs.v + rhs.v); }
fn op_neg(lhs: Value) -> Value { return Value(-lhs.v); }
fn op_sub(lhs: Value, rhs: Value) -> Value { return Value(lhs.v - rhs.v); }
fn op_mul(lhs: Value, rhs: Value) -> Value { return Value(lhs.v * rhs.v); }
fn op_div(lhs: Value, rhs: Value) -> Value { return Value(lhs.v / rhs.v); }
fn op_atan2(lhs: Value, rhs: Value) -> Value { return Value(atan2(lhs.v, rhs.v)); }
fn op_min(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value { return Value(min(lhs.v, rhs.v)); }
fn op_max(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value { return Value(max(lhs.v, rhs.v)); }
fn op_square(lhs: Value) -> Value { return Value(lhs.v * lhs.v); }
fn op_sqrt(lhs: Value) -> Value { return Value(sqrt(lhs.v)); }

fn op_compare(lhs: Value, rhs: Value) -> Value {
    if lhs.v < rhs.v {
        return Value(-1.0);
    } else if lhs.v > rhs.v {
        return Value(1.0);
    } else if lhs.v == rhs.v {
        return Value(0.0);
    } else {
        return Value(nan_f32());
    }
}

fn op_mix(lhs: Value, rhs: Value) -> Value {
    return Value(bitcast<f32>(mix_u32(bitcast<u32>(lhs.v), bitcast<u32>(rhs.v))));
}

fn op_and(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v == 0.0 {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_or(lhs: Value, rhs: Value, stack: ptr<function, Stack>) -> Value {
    if lhs.v != 0.0 {
        return lhs;
    } else {
        return rhs;
    }
}

fn op_not(lhs: Value) -> Value { return Value(f32(lhs.v == 0.0)); }
fn op_rand(lhs: Value) -> Value { return Value(rand(bitcast<u32>(lhs.v))); }
fn op_mod(lhs: Value, rhs: Value) -> Value { return Value(rem_euclid(lhs.v, rhs.v)); }

// --- stack.wgsl ---
const STACK_SIZE_WORDS: u32 = 32u;
const STACK_SIZE_BITS: u32 = STACK_SIZE_WORDS * 32u;
const STACK_SIZE_ITEMS: u32 = STACK_SIZE_WORDS * 16u;

struct Stack {
    offset: u32,
    valid_count: u32,
    has_choice: bool,
    data: array<u32, STACK_SIZE_WORDS>,
}

fn new_stack() -> Stack {
    var data: array<u32, STACK_SIZE_WORDS>;
    for (var i = 0u; i < STACK_SIZE_WORDS; i = i + 1u) {
        data[i] = 0u;
    }
    return Stack(0u, 0u, false, data);
}

fn stack_push(s: ptr<function, Stack>, v: u32) {
    let word_offset = (*s).offset % 32u;
    let word_index = (*s).offset / 32u;
    let mask = 3u << word_offset;
    (*s).data[word_index] &= ~mask;
    (*s).data[word_index] |= (v & 3u) << word_offset;
    (*s).offset = ((*s).offset + 2u) % STACK_SIZE_BITS;
    (*s).valid_count = min((*s).valid_count + 1u, STACK_SIZE_ITEMS);
}

fn stack_pop(s: ptr<function, Stack>) -> u32 {
    if (*s).valid_count == 0u {
        return 3u;
    }
    (*s).valid_count -= 1u;
    (*s).offset = ((*s).offset + STACK_SIZE_BITS - 2u) % STACK_SIZE_BITS;
    let word_offset = (*s).offset % 32u;
    let word_index = (*s).offset / 32u;
    return ((*s).data[word_index] >> word_offset) & 3u;
}

// --- tape_interpreter.wgsl ---
struct TapeResult {
    value: Value,
    pos: u32,
    count: u32,
}

fn run_tape(start: u32, xyz: array<Value, 3>, axes: vec3u) -> TapeResult {
    var i: u32 = start;
    var count: u32 = 0u;
    var reg: array<Value, REG_COUNT>;
    var stack = new_stack();

    var lhs = Value(0.0);
    var rhs = Value(0.0);
    var out = TapeResult(build_imm(nan_f32()), 0u, 0u);
    while true {
        count += 1u;
        let word = tape_data[i];
        let op = unpack4xU8(word.op);
        let rhs_i = op[3];
        let lhs_i = op[2];
        let imm_u = word.imm;
        let imm_v = build_imm(bitcast<f32>(imm_u));
        if lhs_i == 255u {
            lhs = imm_v;
        } else {
            lhs = reg[lhs_i];
        }
        if rhs_i == 255u {
            rhs = imm_v;
        } else {
            rhs = reg[rhs_i];
        }
        var tmp = build_imm(0.0);
        i = i + 1u;
        switch op[0] {
            case OP_OUTPUT: {
                out.value = reg[op[1]];
                continue;
            }
            case OP_INPUT: {
                if imm_u == axes.x {
                    tmp = xyz[0u];
                } else if imm_u == axes.y {
                    tmp = xyz[1u];
                } else if imm_u == axes.z {
                    tmp = xyz[2u];
                } else {
                    tmp = build_imm(var_values[imm_u]);
                }
            }
            case OP_COPY:    { tmp = lhs; }
            case OP_NEG:     { tmp = op_neg(lhs); }
            case OP_ABS:     { tmp = op_abs(lhs); }
            case OP_RECIP:   { tmp = op_recip(lhs); }
            case OP_SQRT:    { tmp = op_sqrt(lhs); }
            case OP_SQUARE:  { tmp = op_square(lhs); }
            case OP_FLOOR:   { tmp = op_floor(lhs); }
            case OP_CEIL:    { tmp = op_ceil(lhs); }
            case OP_ROUND:   { tmp = op_round(lhs); }
            case OP_SIN:     { tmp = op_sin(lhs); }
            case OP_COS:     { tmp = op_cos(lhs); }
            case OP_TAN:     { tmp = op_tan(lhs); }
            case OP_ASIN:    { tmp = op_asin(lhs); }
            case OP_ACOS:    { tmp = op_acos(lhs); }
            case OP_ATAN:    { tmp = op_atan(lhs); }
            case OP_EXP:     { tmp = op_exp(lhs); }
            case OP_LN:      { tmp = op_log(lhs); }
            case OP_NOT:     { tmp = op_not(lhs); }
            case OP_RAND:    { tmp = op_rand(lhs); }
            case OP_ADD:     { tmp = op_add(lhs, rhs); }
            case OP_MUL:     { tmp = op_mul(lhs, rhs); }
            case OP_DIV:     { tmp = op_div(lhs, rhs); }
            case OP_SUB:     { tmp = op_sub(lhs, rhs); }
            case OP_COMPARE: { tmp = op_compare(lhs, rhs); }
            case OP_ATAN2:   { tmp = op_atan2(lhs, rhs); }
            case OP_MOD:     { tmp = op_mod(lhs, rhs); }
            case OP_MIX:     { tmp = op_mix(lhs, rhs); }
            case OP_MIN:     { tmp = op_min(lhs, rhs, &stack); }
            case OP_MAX:     { tmp = op_max(lhs, rhs, &stack); }
            case OP_AND:     { tmp = op_and(lhs, rhs, &stack); }
            case OP_OR:      { tmp = op_or(lhs, rhs, &stack); }
            case OP_MEM: {
                return out;
            }
            case OP_JUMP: {
                if imm_u == 0xFFFFFFFFu {
                    out.pos = i;
                    out.count = count;
                    return out;
                } else if imm_u == 0u {
                    continue;
                } else {
                    i = imm_u;
                    continue;
                }
            }
            default: {
                return out;
            }
        }
        reg[op[1]] = tmp;
    }
    return out;
}
