use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};
use inkwell::types::{BasicTypeEnum, StructType};
use inkwell::values::{BasicValueEnum, PointerValue};

#[derive(Clone)]
enum ColumnType {
    I64,
    F64,
    Bool,
}

struct ColumnDefinition {
    name: String,
    typ: ColumnType,
}

struct TableSchema {
    columns: Vec<ColumnDefinition>,
}

impl TableSchema {
    fn get_column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }
}

enum Column {
    Named(String),
}

enum Expr {
    Column(Column),
    ConstI64(i64),
    Compare(IntPredicate, Box<Expr>, Box<Expr>),
}

impl Expr {
    fn column(name: &str) -> Self {
        Expr::Column(Column::Named(name.to_string()))
    }

    fn const_i64(val: i64) -> Self {
        Expr::ConstI64(val)
    }

    fn compare(pred: IntPredicate, left: Box<Expr>, right: Box<Expr>) -> Self {
        Expr::Compare(pred, left, right)
    }
}

enum AggregateOp {
    Sum,
    Count,
}

struct Query {
    select_aggregate: (AggregateOp, Column),
    where_clause: Option<Expr>,
}

struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

type QueryFunc = unsafe extern "C" fn(*const u8, i64) -> i64;

impl<'ctx> CodeGenerator<'ctx> {
    fn create_row_type(&self, schema: &TableSchema) -> StructType<'ctx> {
        let fields_types: Vec<BasicTypeEnum> = schema.columns.iter()
            .map(|col| match col.typ {
                ColumnType::I64 => self.context.i64_type().into(),
                ColumnType::F64 => self.context.f64_type().into(),
                ColumnType::Bool => self.context.bool_type().into(),
            })
            .collect();

        self.context.struct_type(&fields_types,false)
    }

    fn load_column(
        &self,
        row_ptr: PointerValue<'ctx>,
        row_type: StructType<'ctx>,
        schema: &TableSchema,
        col_name: &str
    ) -> Option<BasicValueEnum<'ctx>> {
        let field_idx = schema.get_column_index(col_name).expect("Column not found");

        let field_ptr = self.builder.build_struct_gep(row_type, row_ptr, field_idx as u32, "field").ok()?;

        let col_def = &schema.columns[field_idx];

        // Load based on column value "type"
        Some(match col_def.typ {
            ColumnType::I64 => self.builder.build_load(self.context.i64_type(), field_ptr, "column").ok()?,
            ColumnType::F64 => self.builder.build_load(self.context.f64_type(), field_ptr, "column").ok()?,
            ColumnType::Bool => self.builder.build_load(self.context.bool_type(), field_ptr, "column").ok()?,
        })
    }

    fn compile_expr(
        &self,
        expr: &Expr,
        row_ptr: PointerValue<'ctx>,
        row_type: StructType<'ctx>,
        schema: &TableSchema,
    ) -> Option<BasicValueEnum<'ctx>> {
        match expr {
            Expr::Column(Column::Named(name)) => {
                self.load_column(row_ptr, row_type, schema, name)
            },
            Expr::ConstI64(val) => {
                Some(self.context.i64_type().const_int(*val as u64, false).into())
            },
            Expr::Compare(pred, left, right) => {
                let lhs = self.compile_expr(left, row_ptr, row_type, schema)?.into_int_value();
                let rhs = self.compile_expr(right, row_ptr, row_type, schema)?.into_int_value();
                Some(self.builder.build_int_compare(*pred, lhs, rhs, "cmp").ok()?.into())
            },
        }
    }

    fn compile_query(&self, query: &Query, schema: &TableSchema) -> Option<JitFunction<'ctx, QueryFunc>> {
        // Signature: fn(*Row, i64) -> i64
        let i64_type = self.context.i64_type();
        let row_type = self.create_row_type(schema);
        let row_ptr = self.context.ptr_type(AddressSpace::default());

        // Create the function type.
        // Takes pointer to row array + length, return i64 (aggregated result)
        let fn_type = i64_type.fn_type(&[row_ptr.into(), i64_type.into()], false);
        let function = self.module.add_function("query", fn_type, None);

        // Loop setup (entry block)
        let entry = self.context.append_basic_block(function, "entry");
        let loop_header = self.context.append_basic_block(function, "loop");
        let loop_body = self.context.append_basic_block(function, "body");
        let loop_exit = self.context.append_basic_block(function, "exit");

        self.builder.position_at_end(entry);

        // Accumulator for results (the running sum)
        let result_ptr = self.builder.build_alloca(i64_type, "result").ok()?;
        self.builder.build_store(result_ptr, i64_type.const_int(0, false)).ok()?;

        // Loop counter (`i`)
        let i_ptr = self.builder.build_alloca(i64_type, "i").ok()?;
        self.builder.build_store(i_ptr, i64_type.const_int(0, false)).ok()?;

        self.builder.build_unconditional_branch(loop_header).ok()?;

        // Loop condition: `while i < len`
        self.builder.position_at_end(loop_header);

        let i = self.builder.build_load(i64_type, i_ptr, "i").ok()?.into_int_value();
        let len = function.get_nth_param(1)?.into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::ULT, i, len, "cond").ok()?;

        self.builder.build_conditional_branch(cond, loop_body, loop_exit).ok()?;

        // Load current row (loop body start)
        self.builder.position_at_end(loop_body);

        // Get pointer to rows[i], `GEP` -> "get element pointer" (LLVM array indexing)
        let rows = function.get_nth_param(0)?.into_pointer_value();
        let row_ptr = unsafe {
            self.builder.build_gep(row_type, rows, &[i], "row_ptr").ok()?
        };

        // Evaluate WHERE clause, if present
        let continue_block = if let Some(where_expr) = &query.where_clause {
            // Compile the expression tree into LLVM IR
            let matches = self.compile_expr(where_expr, row_ptr, row_type, schema)?
                .into_int_value();

            let then_block = self.context.append_basic_block(function, "then");
            let continue_block = self.context.append_basic_block(function, "continue");

            // if (matches) goto then_block else goto continue_block
            self.builder.build_conditional_branch(matches, then_block, continue_block).ok()?;
            self.builder.position_at_end(then_block);

            continue_block
        } else {
            // No WHERE clause, always execute aggregation
            // Still need block for loop structure
            let continue_block = self.context.append_basic_block(function, "continue");

            continue_block
        };

        let (agg_op, select_column) = &query.select_aggregate;

        let Column::Named(col_name) = select_column;
        let select_val = self.load_column(row_ptr, row_type, schema, col_name)?
            .into_int_value();

        // Perform aggregation operation
        match agg_op {
            AggregateOp::Sum => {
                let current_result = self.builder.build_load(i64_type, result_ptr, "current").ok()?.into_int_value();
                let new_result = self.builder.build_int_add(current_result, select_val, "new_result").ok()?;
                self.builder.build_store(result_ptr, new_result).ok()?;
            }
            AggregateOp::Count => {
                let current_result = self.builder.build_load(i64_type, result_ptr, "current").ok()?.into_int_value();
                let new_result = self.builder.build_int_add(current_result, i64_type.const_int(1, false), "inc").ok()?;
                self.builder.build_store(result_ptr, new_result).ok()?;
            }
        }

        self.builder.build_unconditional_branch(continue_block).ok()?;

        // Loop increment
        // `i++`, jump back to the loop condition
        self.builder.position_at_end(continue_block);

        let i_val = self.builder.build_load(i64_type, i_ptr, "i").ok()?.into_int_value();
        let next_i = self.builder.build_int_add(i_val, i64_type.const_int(1, false), "next").ok()?;
        self.builder.build_store(i_ptr, next_i).ok()?;

        self.builder.build_unconditional_branch(loop_header).ok()?;

        // Return result
        // Load accumulated result, then return it
        self.builder.position_at_end(loop_exit);

        let final_result = self.builder.build_load(i64_type, result_ptr, "result").ok()?.into_int_value();
        self.builder.build_return(Some(&final_result)).ok()?;

        unsafe { self.execution_engine.get_function("query").ok() }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::create();
    let module = context.create_module("jit");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive)?;
    let codegen = CodeGenerator {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let schema = TableSchema {
        columns: vec![
            ColumnDefinition { name: "id".into(), typ: ColumnType::I64 },
            ColumnDefinition { name: "value".into(), typ: ColumnType::I64 },
        ]
    };

    // Define runtime data, this is not the schema definition, but must match it
    #[repr(C)]
    struct Row { id: i64, value: i64 }

    let rows = vec![
        Row { id: 1, value: 100 },
        Row { id: 2, value: 200 },
        Row { id: 3, value: 300 },
    ];

    // Build query
    let query = Query {
        select_aggregate: (AggregateOp::Sum, Column::Named("value".into())),
        where_clause: Some(Expr::Compare(
            IntPredicate::SGT,
            Box::from(Expr::column("id")),
            Box::from(Expr::const_i64(1)),
        )),
    };

    let compiled = codegen.compile_query(&query, &schema);

    let result = unsafe { compiled.unwrap().call(rows.as_ptr() as *const u8, rows.len() as i64) };

    println!("SELECT SUM(value) FROM table WHERE id > 1");
    println!("Result: {}", result);

    Ok(())
}
