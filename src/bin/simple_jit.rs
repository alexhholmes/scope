use inkwell::context::Context;
use inkwell::OptimizationLevel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::create();
    let module = context.create_module("jit");
    let builder = context.create_builder();

    let i64_type = context.i64_type();
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
    let function = module.add_function("foo", fn_type, None);

    // Basic block and IR generation

    let basic_block = context.append_basic_block(function, "entry");
    builder.position_at_end(basic_block);

    let x = function.get_nth_param(0).unwrap().into_int_value();
    let y = function.get_nth_param(1).unwrap().into_int_value();

    let two = i64_type.const_int(2, false);
    let mul = builder.build_int_mul(y, two, "mul")?;
    let add = builder.build_int_add(x, mul, "add")?;

    builder.build_return(Some(&add))?;

    // Verification

    function.verify(true);

    // JIT execution

    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;

    let result: i64;
    unsafe {
        let compiled_fn = execution_engine
            .get_function::<unsafe extern "C" fn(i64, i64) -> i64>("foo")?;

        result = compiled_fn.call(5, 3);
    }

    println!("func def foo(x, y) {{ x + y * 2 }}");
    println!("foo(5, 3)");
    println!("Result: {}", result);

    Ok(())
}
