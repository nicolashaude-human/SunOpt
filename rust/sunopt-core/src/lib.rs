// rust/sunopt-core/src/lib.rs

use std::{
    collections::hash_map::DefaultHasher,
    hash::{
        Hash,
        Hasher,
    },
    path::Path,
};

use cranelift::{
    codegen::ir::{
        condcodes::{
            FloatCC,
            IntCC,
        },
        types,
        AbiParam,
        InstBuilder,
        MemFlagsData,
    },
    frontend::{
        FunctionBuilder,
        FunctionBuilderContext,
    },
    jit::{
        JITBuilder,
        JITModule,
    },
    module::{
        default_libcall_names,
        Linkage,
        Module,
    },
};

use oxc::{
    allocator::Allocator,
    ast::ast::{
        LabeledStatement,
        Statement,
    },
    ast_visit::{
        walk::walk_labeled_statement,
        Visit,
    },
    parser::{
        Parser,
        ParserReturn,
    },
    span::SourceType,
};


/*
 * ============================================================
 * CONSTANTS
 * ============================================================
 */

const ENERGY_EPSILON: f64 = 1e-6;

const OUTPUT_COUNT: usize = 11;

const OUT_KWH_NO_SUNSHINE: i32 = 0;
const OUT_BUCKET_DEMAND: i32 = 1;
const OUT_BUCKET_COOLING: i32 = 2;
const OUT_PRE_BUCKET_HOLDING: i32 = 3;
const OUT_ALLOCATED_PV: i32 = 4;
const OUT_UNAVOIDABLE_DEFICIT: i32 = 5;
const OUT_REQUIRED_START: i32 = 6;
const OUT_ADDITIONAL_START: i32 = 7;
const OUT_CURRENT_WATERLINE: i32 = 8;
const OUT_CURRENT_HP_EL: i32 = 9;
const OUT_CURRENT_HP_TH: i32 = 10;


/*
 * 48 h causal-search output slots.
 */
const CAUSAL_OUTPUT_COUNT: usize = 5;

const OUT_CAUSAL_CUT_LEVEL: i32 = 0;
const OUT_CAUSAL_NORMAL_SURVIVES: i32 = 1;
const OUT_CAUSAL_MAX_PV_SURVIVES: i32 = 2;
const OUT_CAUSAL_SURVIVAL_CHECKS: i32 = 3;
const OUT_CAUSAL_BINARY_ITERATIONS: i32 = 4;


/*
 * ============================================================
 * PUBLIC DATA TYPES
 * ============================================================
 */

#[derive(
    Debug,
    Clone,
    PartialEq,
)]
pub struct RustPayloadScope {
    pub label: String,
    pub body_kind: String,
    pub statement_count: usize,
    pub first_statement_kind: Option<String>,
}


#[derive(
    Debug,
    Clone,
    Copy,
)]
pub struct NoSunshineKernelParameters {
    pub current_index: usize,

    pub horizon_steps: usize,
    pub protected_steps: usize,

    pub time_step_hours: f64,

    pub household_reserve_power: f64,

    pub maximum_heat_pump_electrical_power: f64,

    pub cop: f64,

    pub initial_waterline_power: f64,

    pub target_storage_energy: f64,

    /*
     * kWh/K.
     *
     * Example:
     *
     * volume * waterDensity * specificHeat / 3600.
     */
    pub storage_energy_per_kelvin: f64,

    /*
     * W/K.
     *
     * Used when daily_heat_loss < 0.
     */
    pub storage_conductance_w_per_kelvin: f64,

    pub storage_min_temperature: f64,

    pub ambient_temperature: f64,

    /*
     * kWh/day.
     *
     * Set to a negative value to use the insulation /
     * conductance model instead.
     */
    pub daily_heat_loss: f64,

    pub current_storage_energy: f64,
}


#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
)]
pub struct NoSunshineKernelResult {
    pub kwh_no_sunshine: f64,

    pub bucket_demand_energy: f64,

    pub bucket_cooling_loss_energy: f64,

    pub pre_bucket_holding_loss_energy: f64,

    pub allocated_pv_thermal_energy: f64,

    pub unavoidable_thermal_deficit: f64,

    pub required_storage_energy_at_start: f64,

    pub additional_storage_energy_needed_at_start: f64,

    pub current_waterline_power: f64,

    pub current_scheduled_hp_electrical_power: f64,

    pub current_scheduled_hp_thermal_power: f64,
}


pub struct CompiledNoSunshineKernel {
    /*
     * JITModule owns the executable memory.
     *
     * Keep it alive for as long as code_pointer is used.
     */
    _module: JITModule,

    code_pointer: *const u8,

    source_hash: u64,
}


impl CompiledNoSunshineKernel {

    pub fn source_hash(
        &self,
    ) -> u64 {
        self.source_hash
    }


    pub fn execute(
        &self,
        pv_power: &[f64],
        load_energy: &[f64],
        parameters: NoSunshineKernelParameters,
    ) -> Result<NoSunshineKernelResult, String> {

        validate_kernel_inputs(
            pv_power,
            load_energy,
            parameters,
        )?;


        let mut output =
            [0.0_f64; OUTPUT_COUNT];


        type JitFunction =
            unsafe extern "C" fn(
                *const f64,
                *const f64,

                i64,
                i64,
                i64,
                i64,

                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,

                *mut f64,
            );


        let function:
            JitFunction =
            unsafe {
                std::mem::transmute(
                    self.code_pointer,
                )
            };


        unsafe {
            function(
                pv_power.as_ptr(),
                load_energy.as_ptr(),

                pv_power.len() as i64,

                parameters.current_index as i64,

                parameters.horizon_steps as i64,

                parameters.protected_steps as i64,

                parameters.time_step_hours,

                parameters.household_reserve_power,

                parameters
                    .maximum_heat_pump_electrical_power,

                parameters.cop,

                parameters.initial_waterline_power,

                parameters.target_storage_energy,

                parameters.storage_energy_per_kelvin,

                parameters
                    .storage_conductance_w_per_kelvin,

                parameters.storage_min_temperature,

                parameters.ambient_temperature,

                parameters.daily_heat_loss,

                parameters.current_storage_energy,

                output.as_mut_ptr(),
            );
        }


        Ok(
            NoSunshineKernelResult {
                kwh_no_sunshine:
                    output[
                        OUT_KWH_NO_SUNSHINE as usize
                    ],

                bucket_demand_energy:
                    output[
                        OUT_BUCKET_DEMAND as usize
                    ],

                bucket_cooling_loss_energy:
                    output[
                        OUT_BUCKET_COOLING as usize
                    ],

                pre_bucket_holding_loss_energy:
                    output[
                        OUT_PRE_BUCKET_HOLDING as usize
                    ],

                allocated_pv_thermal_energy:
                    output[
                        OUT_ALLOCATED_PV as usize
                    ],

                unavoidable_thermal_deficit:
                    output[
                        OUT_UNAVOIDABLE_DEFICIT as usize
                    ],

                required_storage_energy_at_start:
                    output[
                        OUT_REQUIRED_START as usize
                    ],

                additional_storage_energy_needed_at_start:
                    output[
                        OUT_ADDITIONAL_START as usize
                    ],

                current_waterline_power:
                    output[
                        OUT_CURRENT_WATERLINE as usize
                    ],

                current_scheduled_hp_electrical_power:
                    output[
                        OUT_CURRENT_HP_EL as usize
                    ],

                current_scheduled_hp_thermal_power:
                    output[
                        OUT_CURRENT_HP_TH as usize
                    ],
            },
        )
    }
}



/*
 * ============================================================
 * 48 h PEAK-SHAVING CAUSAL KERNEL TYPES
 * ============================================================
 */

#[derive(
    Debug,
    Clone,
    Copy,
)]
pub struct PeakShavingCausalKernelParameters {
    pub current_index: usize,

    pub horizon_steps: usize,

    pub current_storage_energy: f64,

    pub minimum_reserve_energy: f64,

    pub target_storage_energy: f64,

    pub maximum_heat_pump_electrical_power: f64,

    pub cop: f64,

    pub energy_cut_level: f64,

    pub time_step_hours: f64,

    pub storage_energy_per_kelvin: f64,

    pub storage_conductance_w_per_kelvin: f64,

    pub storage_min_temperature: f64,

    pub ambient_temperature: f64,

    pub daily_heat_loss: f64,
}


#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
)]
pub struct PeakShavingCausalKernelResult {
    pub cut_level_power: f64,

    pub normal_plan_survives: bool,

    pub maximum_pv_plan_survives: bool,

    pub survival_checks: f64,

    pub binary_search_iterations: f64,
}


pub struct CompiledPeakShavingCausalKernel {
    /*
     * JITModule owns the executable memory.
     */
    _module: JITModule,

    code_pointer: *const u8,

    source_hash: u64,
}


impl CompiledPeakShavingCausalKernel {

    pub fn source_hash(
        &self,
    ) -> u64 {
        self.source_hash
    }


    pub fn execute(
        &self,

        export_power: &[f64],

        demand_energy: &[f64],

        parameters:
            PeakShavingCausalKernelParameters,
    ) -> Result<
        PeakShavingCausalKernelResult,
        String,
    > {

        validate_peak_shaving_causal_inputs(
            export_power,
            demand_energy,
            parameters,
        )?;


        let mut output =
            [0.0_f64; CAUSAL_OUTPUT_COUNT];


        type JitFunction =
            unsafe extern "C" fn(
                *const f64,
                *const f64,

                i64,
                i64,
                i64,

                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,
                f64,

                *mut f64,
            );


        let function:
            JitFunction =
            unsafe {
                std::mem::transmute(
                    self.code_pointer,
                )
            };


        unsafe {
            function(
                export_power.as_ptr(),
                demand_energy.as_ptr(),

                export_power.len() as i64,

                parameters.current_index as i64,

                parameters.horizon_steps as i64,

                parameters.current_storage_energy,

                parameters.minimum_reserve_energy,

                parameters.target_storage_energy,

                parameters
                    .maximum_heat_pump_electrical_power,

                parameters.cop,

                parameters.energy_cut_level,

                parameters.time_step_hours,

                parameters.storage_energy_per_kelvin,

                parameters
                    .storage_conductance_w_per_kelvin,

                parameters.storage_min_temperature,

                parameters.ambient_temperature,

                parameters.daily_heat_loss,

                output.as_mut_ptr(),
            );
        }


        Ok(
            PeakShavingCausalKernelResult {
                cut_level_power:
                    output[
                        OUT_CAUSAL_CUT_LEVEL as usize
                    ],

                normal_plan_survives:
                    output[
                        OUT_CAUSAL_NORMAL_SURVIVES as usize
                    ] >
                    0.5,

                maximum_pv_plan_survives:
                    output[
                        OUT_CAUSAL_MAX_PV_SURVIVES as usize
                    ] >
                    0.5,

                survival_checks:
                    output[
                        OUT_CAUSAL_SURVIVAL_CHECKS as usize
                    ],

                binary_search_iterations:
                    output[
                        OUT_CAUSAL_BINARY_ITERATIONS as usize
                    ],
            },
        )
    }
}


pub enum CompiledAccelerationScope {
    NoSunshine(
        CompiledNoSunshineKernel,
    ),

    PeakShavingCausal(
        CompiledPeakShavingCausalKernel,
    ),
}


impl CompiledAccelerationScope {

    pub fn source_hash(
        &self,
    ) -> u64 {

        match self {
            Self::NoSunshine(kernel) =>
                kernel.source_hash(),

            Self::PeakShavingCausal(kernel) =>
                kernel.source_hash(),
        }
    }
}


/*
 * ============================================================
 * TYPESCRIPT / OXC
 * ============================================================
 */

struct RustPayloadVisitor {
    scopes: Vec<RustPayloadScope>,
}


impl RustPayloadVisitor {

    fn new() -> Self {
        Self {
            scopes: Vec::new(),
        }
    }
}


fn statement_kind(
    statement: &Statement<'_>,
) -> &'static str {

    match statement {
        Statement::BlockStatement(_) =>
            "BlockStatement",

        Statement::ForStatement(_) =>
            "ForStatement",

        Statement::IfStatement(_) =>
            "IfStatement",

        Statement::ExpressionStatement(_) =>
            "ExpressionStatement",

        Statement::VariableDeclaration(_) =>
            "VariableDeclaration",

        Statement::ThrowStatement(_) =>
            "ThrowStatement",

        Statement::ReturnStatement(_) =>
            "ReturnStatement",

        _ =>
            "OtherStatement",
    }
}


impl<'a> Visit<'a>
    for RustPayloadVisitor {

    fn visit_labeled_statement(
        &mut self,
        statement: &LabeledStatement<'a>,
    ) {

        let label =
            statement.label.name.as_str();


        if label.starts_with(
            "rust_",
        ) {

            match &statement.body {

                Statement::BlockStatement(block) => {

                    let first_statement_kind =
                        block.body.first().map(
                            |first| {
                                statement_kind(
                                    first,
                                )
                                .to_string()
                            },
                        );


                    self.scopes.push(
                        RustPayloadScope {
                            label:
                                label.to_string(),

                            body_kind:
                                "BlockStatement"
                                    .to_string(),

                            statement_count:
                                block.body.len(),

                            first_statement_kind,
                        },
                    );
                }


                other => {

                    self.scopes.push(
                        RustPayloadScope {
                            label:
                                label.to_string(),

                            body_kind:
                                statement_kind(
                                    other,
                                )
                                .to_string(),

                            statement_count:
                                1,

                            first_statement_kind:
                                Some(
                                    statement_kind(
                                        other,
                                    )
                                    .to_string(),
                                ),
                        },
                    );
                }
            }
        }


        walk_labeled_statement(
            self,
            statement,
        );
    }
}


struct ScopeShapeVisitor<'scope> {
    scope_id: &'scope str,

    result:
        Option<
            Result<
                (),
                String,
            >
        >,
}


impl<'scope>
    ScopeShapeVisitor<'scope> {

    fn new(
        scope_id: &'scope str,
    ) -> Self {

        Self {
            scope_id,

            result: None,
        }
    }
}


impl<'a, 'scope>
    Visit<'a>
    for ScopeShapeVisitor<'scope> {

    fn visit_labeled_statement(
        &mut self,
        statement: &LabeledStatement<'a>,
    ) {

        if self.result.is_some() {
            return;
        }


        if statement.label.name.as_str() ==
            self.scope_id
        {

            self.result =
                Some(
                    validate_scope_shape(
                        self.scope_id,
                        statement,
                    ),
                );

            return;
        }


        walk_labeled_statement(
            self,
            statement,
        );
    }
}


fn validate_scope_shape(
    scope_id: &str,

    statement:
        &LabeledStatement<'_>,
) -> Result<(), String> {

    let Statement::BlockStatement(scope_block) =
        &statement.body
    else {
        return Err(
            format!(
                "{scope_id} must contain a block."
            ),
        );
    };


    if scope_block.body.is_empty() {
        return Err(
            format!(
                "{scope_id} must not be empty."
            ),
        );
    }


    /*
     * Preserve the original 7-day contract.
     *
     * The peak-shaving scope contains the complete causal
     * search and therefore intentionally has more than one
     * top-level statement.
     */
    if scope_id ==
        "rust_noSunshine"
    {

        if scope_block.body.len() != 1 {
            return Err(
                "rust_noSunshine must contain exactly one \
                 top-level statement."
                    .to_string(),
            );
        }


        let Some(
            Statement::ForStatement(for_statement),
        ) =
            scope_block.body.first()
        else {
            return Err(
                "rust_noSunshine must contain a for-loop."
                    .to_string(),
            );
        };


        if !matches!(
            &for_statement.body,
            Statement::BlockStatement(_)
        ) {
            return Err(
                "The rust_noSunshine for-loop must use \
                 a block body."
                    .to_string(),
            );
        }
    }


    Ok(())
}


fn parse_typescript<'a>(
    allocator: &'a Allocator,
    source: &'a str,
    file_path: &Path,
) -> Result<
    oxc::ast::ast::Program<'a>,
    String,
> {

    let source_type =
        SourceType::from_path(
            file_path,
        )
        .map_err(
            |error| {
                format!(
                    "Could not determine source type: \
                     {error:?}"
                )
            },
        )?;


    let ParserReturn {
        program,
        diagnostics,
        fatal_error,
        ..
    } =
        Parser::new(
            allocator,
            source,
            source_type,
        )
        .parse();


    if fatal_error ||
        !diagnostics.is_empty()
    {

        let message =
            diagnostics
                .iter()
                .map(
                    |diagnostic| {
                        format!(
                            "{diagnostic:?}"
                        )
                    },
                )
                .collect::<Vec<_>>()
                .join(
                    "\n",
                );


        return Err(
            format!(
                "Oxc could not parse TypeScript:\n\
                 {message}"
            ),
        );
    }


    Ok(
        program,
    )
}


pub fn inspect_rust_payloads(
    source: &str,
    file_path: &Path,
) -> Result<
    Vec<RustPayloadScope>,
    String,
> {

    let allocator =
        Allocator::default();


    let program =
        parse_typescript(
            &allocator,
            source,
            file_path,
        )?;


    let mut visitor =
        RustPayloadVisitor::new();


    visitor.visit_program(
        &program,
    );


    Ok(
        visitor.scopes,
    )
}


pub fn acceleration_scope_cache_key(
    source: &str,

    scope_id: &str,
) -> Result<u64, String> {

    if !scope_id.starts_with(
        "rust_",
    ) {
        return Err(
            format!(
                "Acceleration scope ID must start with rust_: {scope_id}"
            ),
        );
    }


    /*
     * IMPORTANT:
     *
     * This function is used on every controller call.
     * It must NOT parse TypeScript.
     *
     * Oxc runs only on a cache miss inside the compiler path.
     */
    let mut hasher =
        DefaultHasher::new();


    source.hash(
        &mut hasher,
    );

    scope_id.hash(
        &mut hasher,
    );


    Ok(
        hasher.finish(),
    )
}


pub fn acceleration_scope_source_hash(
    source: &str,

    file_path: &Path,

    scope_id: &str,
) -> Result<u64, String> {

    /*
     * Validation/hash helper retained for tests and diagnostics.
     *
     * Unlike acceleration_scope_cache_key(), this DOES parse and
     * validate the requested scope.
     */
    validate_scope_source(
        source,
        file_path,
        scope_id,
    )
}


fn validate_scope_source(
    source: &str,

    file_path: &Path,

    scope_id: &str,
) -> Result<u64, String> {

    if !scope_id.starts_with(
        "rust_",
    ) {
        return Err(
            format!(
                "Acceleration scope ID must start with rust_: {scope_id}"
            ),
        );
    }


    let allocator =
        Allocator::default();


    let program =
        parse_typescript(
            &allocator,
            source,
            file_path,
        )?;


    let mut visitor =
        ScopeShapeVisitor::new(
            scope_id,
        );


    visitor.visit_program(
        &program,
    );


    visitor
        .result
        .ok_or_else(
            || {
                format!(
                    "Could not find acceleration scope {scope_id}."
                )
            },
        )??;


    /*
     * Cache identity is source + scope ID.
     *
     * Two scopes inside the same TypeScript source therefore
     * never collide.
     */
    let mut hasher =
        DefaultHasher::new();


    source.hash(
        &mut hasher,
    );

    scope_id.hash(
        &mut hasher,
    );


    Ok(
        hasher.finish(),
    )
}


/*
 * ============================================================
 * VALIDATION
 * ============================================================
 */

fn validate_kernel_inputs(
    pv_power: &[f64],
    load_energy: &[f64],
    parameters: NoSunshineKernelParameters,
) -> Result<(), String> {

    if pv_power.is_empty() {
        return Err(
            "pv_power must not be empty."
                .to_string(),
        );
    }


    if pv_power.len() !=
        load_energy.len()
    {
        return Err(
            "pv_power and load_energy must have \
             equal length."
                .to_string(),
        );
    }


    if parameters.current_index >=
        pv_power.len()
    {
        return Err(
            "current_index is outside the data."
                .to_string(),
        );
    }


    if parameters.horizon_steps == 0 {
        return Err(
            "horizon_steps must be greater than zero."
                .to_string(),
        );
    }


    if parameters.horizon_steps >
        pv_power.len()
    {
        return Err(
            "horizon_steps cannot exceed data length."
                .to_string(),
        );
    }


    if parameters.protected_steps >
        parameters.horizon_steps
    {
        return Err(
            "protected_steps cannot exceed \
             horizon_steps."
                .to_string(),
        );
    }


    if parameters.time_step_hours <= 0.0 {
        return Err(
            "time_step_hours must be greater than zero."
                .to_string(),
        );
    }


    if parameters.cop <= 0.0 {
        return Err(
            "cop must be greater than zero."
                .to_string(),
        );
    }


    if parameters
        .maximum_heat_pump_electrical_power <
        0.0
    {
        return Err(
            "maximum heat-pump electrical power \
             cannot be negative."
                .to_string(),
        );
    }


    if parameters.target_storage_energy <
        0.0
    {
        return Err(
            "target_storage_energy cannot be negative."
                .to_string(),
        );
    }


    if parameters.storage_energy_per_kelvin <=
        0.0
    {
        return Err(
            "storage_energy_per_kelvin must be \
             greater than zero."
                .to_string(),
        );
    }


    if parameters.daily_heat_loss < 0.0 &&
        parameters
            .storage_conductance_w_per_kelvin <
            0.0
    {
        return Err(
            "storage conductance cannot be negative."
                .to_string(),
        );
    }


    Ok(())
}



fn validate_peak_shaving_causal_inputs(
    export_power: &[f64],

    demand_energy: &[f64],

    parameters:
        PeakShavingCausalKernelParameters,
) -> Result<(), String> {

    if export_power.is_empty() {
        return Err(
            "export_power must not be empty."
                .to_string(),
        );
    }


    if export_power.len() !=
        demand_energy.len()
    {
        return Err(
            "export_power and demand_energy must have equal length."
                .to_string(),
        );
    }


    if parameters.current_index >=
        export_power.len()
    {
        return Err(
            "current_index is outside the data."
                .to_string(),
        );
    }


    if parameters.horizon_steps == 0 {
        return Err(
            "horizon_steps must be greater than zero."
                .to_string(),
        );
    }


    if parameters.horizon_steps >
        export_power.len()
    {
        return Err(
            "horizon_steps cannot exceed data length."
                .to_string(),
        );
    }


    if parameters.time_step_hours <= 0.0 {
        return Err(
            "time_step_hours must be greater than zero."
                .to_string(),
        );
    }


    if parameters.cop <= 0.0 {
        return Err(
            "cop must be greater than zero."
                .to_string(),
        );
    }


    if parameters
        .maximum_heat_pump_electrical_power <
        0.0
    {
        return Err(
            "maximum heat-pump electrical power cannot be negative."
                .to_string(),
        );
    }


    if parameters.target_storage_energy <
        0.0
    {
        return Err(
            "target_storage_energy cannot be negative."
                .to_string(),
        );
    }


    if parameters.storage_energy_per_kelvin <=
        0.0
    {
        return Err(
            "storage_energy_per_kelvin must be greater than zero."
                .to_string(),
        );
    }


    if parameters.daily_heat_loss < 0.0 &&
        parameters
            .storage_conductance_w_per_kelvin <
            0.0
    {
        return Err(
            "storage conductance cannot be negative."
                .to_string(),
        );
    }


    Ok(())
}


/*
 * ============================================================
 * REFERENCE IMPLEMENTATION
 * ============================================================
 *
 * This remains deliberately ordinary Rust.
 *
 * It provides a point-for-point oracle for the Cranelift JIT
 * while we are wiring the native backend.
 * ============================================================
 */

#[derive(
    Debug,
    Clone,
    Copy,
)]
struct ReverseStorageResult {
    required_before: f64,
    cooling_loss: f64,
    unavoidable_deficit: f64,
}


fn rust_max(
    left: f64,
    right: f64,
) -> f64 {

    if left > right {
        left
    } else {
        right
    }
}


fn rust_min(
    left: f64,
    right: f64,
) -> f64 {

    if left < right {
        left
    } else {
        right
    }
}


fn calculate_requested_loss(
    energy: f64,
    parameters: NoSunshineKernelParameters,
) -> f64 {

    if parameters.daily_heat_loss >= 0.0 {

        return parameters.daily_heat_loss *
            parameters.time_step_hours /
            24.0;
    }


    let temperature_delta =
        parameters.storage_min_temperature -
        parameters.ambient_temperature +
        energy /
        parameters.storage_energy_per_kelvin;


    let positive_delta =
        rust_max(
            0.0,
            temperature_delta,
        );


    parameters
        .storage_conductance_w_per_kelvin *
        parameters.time_step_hours /
        1000.0 *
        positive_delta
}


fn calculate_required_before_reference(
    required_after: f64,
    demand: f64,
    parameters: NoSunshineKernelParameters,
) -> ReverseStorageResult {

    let required_sum =
        required_after +
        demand;


    let candidate =
        if parameters.daily_heat_loss >= 0.0 {

            if required_sum <= 0.0 {
                0.0
            } else {
                required_sum +
                parameters.daily_heat_loss *
                parameters.time_step_hours /
                24.0
            }

        } else {

            let q =
                parameters
                    .storage_conductance_w_per_kelvin *
                parameters.time_step_hours /
                1000.0;


            let ambient_delta =
                parameters.storage_min_temperature -
                parameters.ambient_temperature;


            let zero_loss_temperature_delta =
                ambient_delta +
                required_sum /
                parameters.storage_energy_per_kelvin;


            if zero_loss_temperature_delta <= 0.0 {

                required_sum

            } else {

                let denominator =
                    1.0 -
                    q /
                    parameters
                        .storage_energy_per_kelvin;


                (
                    required_sum +
                    q * ambient_delta
                ) /
                denominator
            }
        };


    let capped =
        candidate >
        parameters.target_storage_energy;


    let required_before =
        rust_min(
            candidate,
            parameters.target_storage_energy,
        );


    let requested_loss =
        calculate_requested_loss(
            required_before,
            parameters,
        );


    let cooling_loss =
        rust_min(
            required_before,
            rust_max(
                0.0,
                requested_loss,
            ),
        );


    let after_loss =
        required_before -
        cooling_loss;


    let discharged =
        rust_min(
            demand,
            after_loss,
        );


    let unmet_demand =
        rust_max(
            0.0,
            demand -
            discharged,
        );


    let stored_after =
        after_loss -
        discharged;


    let missing_future =
        rust_max(
            0.0,
            required_after -
            stored_after,
        );


    let unavoidable_deficit =
        if capped {
            unmet_demand +
            missing_future
        } else {
            0.0
        };


    ReverseStorageResult {
        required_before,
        cooling_loss,
        unavoidable_deficit,
    }
}


pub fn run_no_sunshine_reference(
    pv_power: &[f64],
    load_energy: &[f64],
    parameters: NoSunshineKernelParameters,
) -> Result<
    NoSunshineKernelResult,
    String,
> {

    validate_kernel_inputs(
        pv_power,
        load_energy,
        parameters,
    )?;


    let mut required_after =
        0.0;


    let mut bucket_demand_total =
        0.0;

    let mut bucket_cooling_total =
        0.0;

    let mut pre_bucket_holding_loss =
        0.0;

    let mut allocated_pv_total =
        0.0;

    let mut unavoidable_deficit_total =
        0.0;


    let mut current_waterline =
        parameters.initial_waterline_power;

    let mut current_hp_el =
        0.0;


    for offset in
        (0..parameters.horizon_steps).rev()
    {

        let data_index =
            (
                parameters.current_index +
                offset
            ) %
            pv_power.len();


        let pv =
            pv_power[
                data_index
            ];


        let load =
            load_energy[
                data_index
            ];


        let protected =
            offset <
            parameters.protected_steps;


        let bucket_demand =
            if protected {
                0.0
            } else {
                load
            };


        let export_power =
            rust_max(
                0.0,
                pv -
                parameters.household_reserve_power,
            );


        let available_hp_el =
            rust_min(
                export_power,
                parameters
                    .maximum_heat_pump_electrical_power,
            );


        let available_hp_th =
            available_hp_el *
            parameters.cop *
            parameters.time_step_hours;


        let allocated_hp_th =
            rust_min(
                required_after,
                available_hp_th,
            );


        let allocated_hp_el =
            if allocated_hp_th > 0.0 {
                allocated_hp_th /
                (
                    parameters.cop *
                    parameters.time_step_hours
                )
            } else {
                0.0
            };


        allocated_pv_total +=
            allocated_hp_th;


        let inactive_waterline =
            rust_max(
                parameters.initial_waterline_power,
                export_power,
            );


        let waterline =
            if allocated_hp_el >
                ENERGY_EPSILON
            {
                rust_max(
                    0.0,
                    export_power -
                    allocated_hp_el,
                )
            } else {
                inactive_waterline
            };


        let remaining_after_charge =
            rust_max(
                0.0,
                required_after -
                allocated_hp_th,
            );


        let reverse =
            calculate_required_before_reference(
                remaining_after_charge,
                bucket_demand,
                parameters,
            );


        if protected {
            pre_bucket_holding_loss +=
                reverse.cooling_loss;
        } else {
            bucket_demand_total +=
                bucket_demand;

            bucket_cooling_total +=
                reverse.cooling_loss;
        }


        unavoidable_deficit_total +=
            reverse.unavoidable_deficit;


        required_after =
            reverse.required_before;


        if offset == 0 {
            current_waterline =
                waterline;

            current_hp_el =
                allocated_hp_el;
        }
    }


    let kwh_no_sunshine =
        bucket_demand_total +
        bucket_cooling_total;


    let required_start =
        required_after;


    let additional_start =
        rust_max(
            0.0,
            required_start -
            parameters.current_storage_energy,
        );


    Ok(
        NoSunshineKernelResult {
            kwh_no_sunshine,

            bucket_demand_energy:
                bucket_demand_total,

            bucket_cooling_loss_energy:
                bucket_cooling_total,

            pre_bucket_holding_loss_energy:
                pre_bucket_holding_loss,

            allocated_pv_thermal_energy:
                allocated_pv_total,

            unavoidable_thermal_deficit:
                unavoidable_deficit_total,

            required_storage_energy_at_start:
                required_start,

            additional_storage_energy_needed_at_start:
                additional_start,

            current_waterline_power:
                current_waterline,

            current_scheduled_hp_electrical_power:
                current_hp_el,

            current_scheduled_hp_thermal_power:
                current_hp_el *
                parameters.cop,
        },
    )
}


/*
 * ============================================================
 * CRANELIFT HELPERS
 * ============================================================
 */

fn f64_max(
    builder: &mut FunctionBuilder<'_>,
    left: cranelift::codegen::ir::Value,
    right: cranelift::codegen::ir::Value,
) -> cranelift::codegen::ir::Value {

    let condition =
        builder.ins().fcmp(
            FloatCC::GreaterThan,
            left,
            right,
        );


    builder.ins().select(
        condition,
        left,
        right,
    )
}


fn f64_min(
    builder: &mut FunctionBuilder<'_>,
    left: cranelift::codegen::ir::Value,
    right: cranelift::codegen::ir::Value,
) -> cranelift::codegen::ir::Value {

    let condition =
        builder.ins().fcmp(
            FloatCC::LessThan,
            left,
            right,
        );


    builder.ins().select(
        condition,
        left,
        right,
    )
}


fn load_f64_indexed(
    builder: &mut FunctionBuilder<'_>,
    base_pointer: cranelift::codegen::ir::Value,
    index: cranelift::codegen::ir::Value,
) -> cranelift::codegen::ir::Value {

    let byte_offset =
        builder
            .ins()
            .imul_imm_s(
                index,
                8,
            );


    let address =
        builder
            .ins()
            .iadd(
                base_pointer,
                byte_offset,
            );


    builder
        .ins()
        .load(
            types::F64,
            MemFlagsData::new(),
            address,
            0,
        )
}


fn store_output(
    builder: &mut FunctionBuilder<'_>,
    output_pointer: cranelift::codegen::ir::Value,
    index: i32,
    value: cranelift::codegen::ir::Value,
) {

    builder
        .ins()
        .store(
            MemFlagsData::new(),
            value,
            output_pointer,
            index * 8,
        );
}


/*
 * ============================================================
 * CRANELIFT – INLINE REVERSE STORAGE STEP
 * ============================================================
 */

struct ReverseStorageValues {
    required_before:
        cranelift::codegen::ir::Value,

    cooling_loss:
        cranelift::codegen::ir::Value,

    unavoidable_deficit:
        cranelift::codegen::ir::Value,
}


#[allow(
    clippy::too_many_arguments,
)]
fn lower_reverse_storage_step(
    builder: &mut FunctionBuilder<'_>,

    required_after:
        cranelift::codegen::ir::Value,

    demand:
        cranelift::codegen::ir::Value,

    time_step_hours:
        cranelift::codegen::ir::Value,

    target_storage_energy:
        cranelift::codegen::ir::Value,

    energy_per_kelvin:
        cranelift::codegen::ir::Value,

    conductance:
        cranelift::codegen::ir::Value,

    minimum_temperature:
        cranelift::codegen::ir::Value,

    ambient_temperature:
        cranelift::codegen::ir::Value,

    daily_heat_loss:
        cranelift::codegen::ir::Value,

) -> ReverseStorageValues {

    let zero =
        builder
            .ins()
            .f64const(
                0.0,
            );


    let one =
        builder
            .ins()
            .f64const(
                1.0,
            );


    let thousand =
        builder
            .ins()
            .f64const(
                1000.0,
            );


    let twenty_four =
        builder
            .ins()
            .f64const(
                24.0,
            );


    let required_sum =
        builder
            .ins()
            .fadd(
                required_after,
                demand,
            );


    /*
     * --------------------------------------------------------
     * Insulation / conductance candidate
     * --------------------------------------------------------
     */

    let conductance_times_dt =
        builder
            .ins()
            .fmul(
                conductance,
                time_step_hours,
            );


    let q =
        builder
            .ins()
            .fdiv(
                conductance_times_dt,
                thousand,
            );


    let ambient_delta =
        builder
            .ins()
            .fsub(
                minimum_temperature,
                ambient_temperature,
            );


    let energy_temperature_delta =
        builder
            .ins()
            .fdiv(
                required_sum,
                energy_per_kelvin,
            );


    let zero_loss_temperature_delta =
        builder
            .ins()
            .fadd(
                ambient_delta,
                energy_temperature_delta,
            );


    let q_over_capacity =
        builder
            .ins()
            .fdiv(
                q,
                energy_per_kelvin,
            );


    let denominator =
        builder
            .ins()
            .fsub(
                one,
                q_over_capacity,
            );


    let q_times_ambient_delta =
        builder
            .ins()
            .fmul(
                q,
                ambient_delta,
            );


    let positive_loss_numerator =
        builder
            .ins()
            .fadd(
                required_sum,
                q_times_ambient_delta,
            );


    let positive_loss_candidate =
        builder
            .ins()
            .fdiv(
                positive_loss_numerator,
                denominator,
            );


    let has_positive_loss =
        builder
            .ins()
            .fcmp(
                FloatCC::GreaterThan,
                zero_loss_temperature_delta,
                zero,
            );


    let insulation_candidate =
        builder
            .ins()
            .select(
                has_positive_loss,
                positive_loss_candidate,
                required_sum,
            );


    /*
     * --------------------------------------------------------
     * Legacy daily-loss candidate
     * --------------------------------------------------------
     */

    let daily_loss_times_dt =
        builder
            .ins()
            .fmul(
                daily_heat_loss,
                time_step_hours,
            );


    let daily_loss_step =
        builder
            .ins()
            .fdiv(
                daily_loss_times_dt,
                twenty_four,
            );


    let daily_positive_candidate =
        builder
            .ins()
            .fadd(
                required_sum,
                daily_loss_step,
            );


    let has_required_energy =
        builder
            .ins()
            .fcmp(
                FloatCC::GreaterThan,
                required_sum,
                zero,
            );


    let daily_candidate =
        builder
            .ins()
            .select(
                has_required_energy,
                daily_positive_candidate,
                zero,
            );


    let use_daily_loss =
        builder
            .ins()
            .fcmp(
                FloatCC::GreaterThanOrEqual,
                daily_heat_loss,
                zero,
            );


    let candidate =
        builder
            .ins()
            .select(
                use_daily_loss,
                daily_candidate,
                insulation_candidate,
            );


    let capped =
        builder
            .ins()
            .fcmp(
                FloatCC::GreaterThan,
                candidate,
                target_storage_energy,
            );


    let required_before =
        f64_min(
            builder,
            candidate,
            target_storage_energy,
        );


    /*
     * --------------------------------------------------------
     * Actual heat loss at required_before
     * --------------------------------------------------------
     */

    let required_temperature_delta =
        builder
            .ins()
            .fdiv(
                required_before,
                energy_per_kelvin,
            );


    let storage_to_ambient_delta =
        builder
            .ins()
            .fadd(
                ambient_delta,
                required_temperature_delta,
            );


    let positive_storage_delta =
        f64_max(
            builder,
            zero,
            storage_to_ambient_delta,
        );


    let insulation_requested_loss =
        builder
            .ins()
            .fmul(
                q,
                positive_storage_delta,
            );


    let requested_loss =
        builder
            .ins()
            .select(
                use_daily_loss,
                daily_loss_step,
                insulation_requested_loss,
            );


    let positive_requested_loss =
        f64_max(
            builder,
            zero,
            requested_loss,
        );


    let cooling_loss =
        f64_min(
            builder,
            required_before,
            positive_requested_loss,
        );


    let after_loss =
        builder
            .ins()
            .fsub(
                required_before,
                cooling_loss,
            );


    let discharged =
        f64_min(
            builder,
            demand,
            after_loss,
        );


    let unmet_demand_raw =
        builder
            .ins()
            .fsub(
                demand,
                discharged,
            );


    let unmet_demand =
        f64_max(
            builder,
            zero,
            unmet_demand_raw,
        );


    let stored_after =
        builder
            .ins()
            .fsub(
                after_loss,
                discharged,
            );


    let missing_future_raw =
        builder
            .ins()
            .fsub(
                required_after,
                stored_after,
            );


    let missing_future =
        f64_max(
            builder,
            zero,
            missing_future_raw,
        );


    let deficit_when_capped =
        builder
            .ins()
            .fadd(
                unmet_demand,
                missing_future,
            );


    let unavoidable_deficit =
        builder
            .ins()
            .select(
                capped,
                deficit_when_capped,
                zero,
            );


    ReverseStorageValues {
        required_before,
        cooling_loss,
        unavoidable_deficit,
    }
}


/*
 * ============================================================
 * CRANELIFT – COMPLETE 7-DAY FAST KERNEL
 * ============================================================
 */

fn add_kernel_signature(
    module: &JITModule,
) -> cranelift::codegen::ir::Signature {

    let mut signature =
        module.make_signature();


    let pointer_type =
        module
            .target_config()
            .pointer_type();


    /*
     * Arrays.
     */
    signature.params.push(
        AbiParam::new(
            pointer_type,
        ),
    );

    signature.params.push(
        AbiParam::new(
            pointer_type,
        ),
    );


    /*
     * Integer parameters.
     */
    for _ in 0..4 {
        signature.params.push(
            AbiParam::new(
                types::I64,
            ),
        );
    }


    /*
     * Floating-point parameters.
     */
    for _ in 0..12 {
        signature.params.push(
            AbiParam::new(
                types::F64,
            ),
        );
    }


    /*
     * Output array.
     */
    signature.params.push(
        AbiParam::new(
            pointer_type,
        ),
    );


    signature
}


fn build_no_sunshine_function(
    module: &mut JITModule,
) -> Result<
    cranelift::module::FuncId,
    String,
> {

    let pointer_type =
        module
            .target_config()
            .pointer_type();


    if pointer_type != types::I64 {
        return Err(
            "The current SunOpt JIT kernel requires \
             a 64-bit host."
                .to_string(),
        );
    }


    let signature =
        add_kernel_signature(
            module,
        );


    let function_id =
        module
            .declare_function(
                "sunopt_no_sunshine_full",
                Linkage::Local,
                &signature,
            )
            .map_err(
                |error| {
                    format!(
                        "Could not declare no-sunshine \
                         JIT function: {error}"
                    )
                },
            )?;


    let mut context =
        module.make_context();


    context.func.signature =
        signature;


    let target_config =
        module.target_config();


    let mut builder_context =
        FunctionBuilderContext::new();


    {
        let mut builder =
            FunctionBuilder::new(
                &mut context.func,
                &mut builder_context,
            );


        let entry =
            builder.create_block();

        let loop_header =
            builder.create_block();

        let loop_body =
            builder.create_block();

        let exit =
            builder.create_block();


        builder
            .append_block_params_for_function_params(
                entry,
            );


        builder.switch_to_block(
            entry,
        );


        let params =
            builder.block_params(
                entry,
            )
            .to_vec();


        let pv_pointer =
            params[0];

        let load_pointer =
            params[1];

        let data_length =
            params[2];

        let current_index =
            params[3];

        let horizon_steps =
            params[4];

        let protected_steps =
            params[5];

        let time_step_hours =
            params[6];

        let household_reserve_power =
            params[7];

        let maximum_hp_el =
            params[8];

        let cop =
            params[9];

        let initial_waterline =
            params[10];

        let target_storage_energy =
            params[11];

        let energy_per_kelvin =
            params[12];

        let conductance =
            params[13];

        let minimum_temperature =
            params[14];

        let ambient_temperature =
            params[15];

        let daily_heat_loss =
            params[16];

        let current_storage_energy =
            params[17];

        let output_pointer =
            params[18];


        let zero_f64 =
            builder
                .ins()
                .f64const(
                    0.0,
                );


        let one_i64 =
            builder
                .ins()
                .iconst(
                    types::I64,
                    1,
                );


        let zero_i64 =
            builder
                .ins()
                .iconst(
                    types::I64,
                    0,
                );


        let epsilon =
            builder
                .ins()
                .f64const(
                    ENERGY_EPSILON,
                );


        /*
         * Mutable source-language variables.
         *
         * FunctionBuilder converts these into SSA.
         */
        let offset_var =
            builder.declare_var(
                types::I64,
            );

        let required_after_var =
            builder.declare_var(
                types::F64,
            );

        let bucket_demand_total_var =
            builder.declare_var(
                types::F64,
            );

        let bucket_cooling_total_var =
            builder.declare_var(
                types::F64,
            );

        let pre_bucket_holding_var =
            builder.declare_var(
                types::F64,
            );

        let allocated_pv_total_var =
            builder.declare_var(
                types::F64,
            );

        let unavoidable_deficit_var =
            builder.declare_var(
                types::F64,
            );

        let current_waterline_var =
            builder.declare_var(
                types::F64,
            );

        let current_hp_el_var =
            builder.declare_var(
                types::F64,
            );


        let first_offset =
            builder
                .ins()
                .isub(
                    horizon_steps,
                    one_i64,
                );


        builder.def_var(
            offset_var,
            first_offset,
        );

        builder.def_var(
            required_after_var,
            zero_f64,
        );

        builder.def_var(
            bucket_demand_total_var,
            zero_f64,
        );

        builder.def_var(
            bucket_cooling_total_var,
            zero_f64,
        );

        builder.def_var(
            pre_bucket_holding_var,
            zero_f64,
        );

        builder.def_var(
            allocated_pv_total_var,
            zero_f64,
        );

        builder.def_var(
            unavoidable_deficit_var,
            zero_f64,
        );

        builder.def_var(
            current_waterline_var,
            initial_waterline,
        );

        builder.def_var(
            current_hp_el_var,
            zero_f64,
        );


        builder
            .ins()
            .jump(
                loop_header,
                &[],
            );


        builder.seal_block(
            entry,
        );


        /*
         * ----------------------------------------------------
         * LOOP HEADER
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            loop_header,
        );


        let offset =
            builder.use_var(
                offset_var,
            );


        let continue_loop =
            builder
                .ins()
                .icmp(
                    IntCC::SignedGreaterThanOrEqual,
                    offset,
                    zero_i64,
                );


        builder
            .ins()
            .brif(
                continue_loop,
                loop_body,
                &[],
                exit,
                &[],
            );


        /*
         * ----------------------------------------------------
         * LOOP BODY
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            loop_body,
        );

        builder.seal_block(
            loop_body,
        );


        let raw_data_index =
            builder
                .ins()
                .iadd(
                    current_index,
                    offset,
                );


        let data_index =
            builder
                .ins()
                .urem(
                    raw_data_index,
                    data_length,
                );


        let pv_power =
            load_f64_indexed(
                &mut builder,
                pv_pointer,
                data_index,
            );


        let load_energy =
            load_f64_indexed(
                &mut builder,
                load_pointer,
                data_index,
            );


        let protected =
            builder
                .ins()
                .icmp(
                    IntCC::SignedLessThan,
                    offset,
                    protected_steps,
                );


        let bucket_demand =
            builder
                .ins()
                .select(
                    protected,
                    zero_f64,
                    load_energy,
                );


        let pv_minus_household =
            builder
                .ins()
                .fsub(
                    pv_power,
                    household_reserve_power,
                );


        let export_power =
            f64_max(
                &mut builder,
                zero_f64,
                pv_minus_household,
            );


        let available_hp_el =
            f64_min(
                &mut builder,
                export_power,
                maximum_hp_el,
            );


        let hp_times_cop =
            builder
                .ins()
                .fmul(
                    available_hp_el,
                    cop,
                );


        let available_hp_th =
            builder
                .ins()
                .fmul(
                    hp_times_cop,
                    time_step_hours,
                );


        let required_after =
            builder.use_var(
                required_after_var,
            );


        let allocated_hp_th =
            f64_min(
                &mut builder,
                required_after,
                available_hp_th,
            );


        let hp_denominator =
            builder
                .ins()
                .fmul(
                    cop,
                    time_step_hours,
                );


        let allocated_hp_el_positive =
            builder
                .ins()
                .fdiv(
                    allocated_hp_th,
                    hp_denominator,
                );


        let allocated_hp_positive =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThan,
                    allocated_hp_th,
                    zero_f64,
                );


        let allocated_hp_el =
            builder
                .ins()
                .select(
                    allocated_hp_positive,
                    allocated_hp_el_positive,
                    zero_f64,
                );


        let old_allocated_pv_total =
            builder.use_var(
                allocated_pv_total_var,
            );


        let new_allocated_pv_total =
            builder
                .ins()
                .fadd(
                    old_allocated_pv_total,
                    allocated_hp_th,
                );


        builder.def_var(
            allocated_pv_total_var,
            new_allocated_pv_total,
        );


        let inactive_waterline =
            f64_max(
                &mut builder,
                initial_waterline,
                export_power,
            );


        let lowered_waterline_raw =
            builder
                .ins()
                .fsub(
                    export_power,
                    allocated_hp_el,
                );


        let lowered_waterline =
            f64_max(
                &mut builder,
                zero_f64,
                lowered_waterline_raw,
            );


        let waterline_active =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThan,
                    allocated_hp_el,
                    epsilon,
                );


        let waterline =
            builder
                .ins()
                .select(
                    waterline_active,
                    lowered_waterline,
                    inactive_waterline,
                );


        let remaining_after_charge_raw =
            builder
                .ins()
                .fsub(
                    required_after,
                    allocated_hp_th,
                );


        let remaining_after_charge =
            f64_max(
                &mut builder,
                zero_f64,
                remaining_after_charge_raw,
            );


        let reverse =
            lower_reverse_storage_step(
                &mut builder,

                remaining_after_charge,

                bucket_demand,

                time_step_hours,

                target_storage_energy,

                energy_per_kelvin,

                conductance,

                minimum_temperature,

                ambient_temperature,

                daily_heat_loss,
            );


        let old_pre_bucket_holding =
            builder.use_var(
                pre_bucket_holding_var,
            );


        let pre_bucket_candidate =
            builder
                .ins()
                .fadd(
                    old_pre_bucket_holding,
                    reverse.cooling_loss,
                );


        let new_pre_bucket_holding =
            builder
                .ins()
                .select(
                    protected,
                    pre_bucket_candidate,
                    old_pre_bucket_holding,
                );


        builder.def_var(
            pre_bucket_holding_var,
            new_pre_bucket_holding,
        );


        let old_bucket_demand_total =
            builder.use_var(
                bucket_demand_total_var,
            );


        let bucket_demand_candidate =
            builder
                .ins()
                .fadd(
                    old_bucket_demand_total,
                    bucket_demand,
                );


        let new_bucket_demand_total =
            builder
                .ins()
                .select(
                    protected,
                    old_bucket_demand_total,
                    bucket_demand_candidate,
                );


        builder.def_var(
            bucket_demand_total_var,
            new_bucket_demand_total,
        );


        let old_bucket_cooling_total =
            builder.use_var(
                bucket_cooling_total_var,
            );


        let bucket_cooling_candidate =
            builder
                .ins()
                .fadd(
                    old_bucket_cooling_total,
                    reverse.cooling_loss,
                );


        let new_bucket_cooling_total =
            builder
                .ins()
                .select(
                    protected,
                    old_bucket_cooling_total,
                    bucket_cooling_candidate,
                );


        builder.def_var(
            bucket_cooling_total_var,
            new_bucket_cooling_total,
        );


        let old_unavoidable_deficit =
            builder.use_var(
                unavoidable_deficit_var,
            );


        let new_unavoidable_deficit =
            builder
                .ins()
                .fadd(
                    old_unavoidable_deficit,
                    reverse.unavoidable_deficit,
                );


        builder.def_var(
            unavoidable_deficit_var,
            new_unavoidable_deficit,
        );


        builder.def_var(
            required_after_var,
            reverse.required_before,
        );


        /*
         * The current schedule point is offset == 0.
         */
        let is_current_point =
            builder
                .ins()
                .icmp(
                    IntCC::Equal,
                    offset,
                    zero_i64,
                );


        let previous_current_waterline =
            builder.use_var(
                current_waterline_var,
            );


        let current_waterline =
            builder
                .ins()
                .select(
                    is_current_point,
                    waterline,
                    previous_current_waterline,
                );


        builder.def_var(
            current_waterline_var,
            current_waterline,
        );


        let previous_current_hp_el =
            builder.use_var(
                current_hp_el_var,
            );


        let current_hp_el =
            builder
                .ins()
                .select(
                    is_current_point,
                    allocated_hp_el,
                    previous_current_hp_el,
                );


        builder.def_var(
            current_hp_el_var,
            current_hp_el,
        );


        let next_offset =
            builder
                .ins()
                .isub(
                    offset,
                    one_i64,
                );


        builder.def_var(
            offset_var,
            next_offset,
        );


        builder
            .ins()
            .jump(
                loop_header,
                &[],
            );


        /*
         * Both predecessors of loop_header are now known:
         *
         * entry + loop_body.
         */
        builder.seal_block(
            loop_header,
        );


        /*
         * ----------------------------------------------------
         * EXIT
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            exit,
        );

        builder.seal_block(
            exit,
        );


        let bucket_demand_total =
            builder.use_var(
                bucket_demand_total_var,
            );


        let bucket_cooling_total =
            builder.use_var(
                bucket_cooling_total_var,
            );


        let kwh_no_sunshine =
            builder
                .ins()
                .fadd(
                    bucket_demand_total,
                    bucket_cooling_total,
                );


        let required_start =
            builder.use_var(
                required_after_var,
            );


        let additional_start_raw =
            builder
                .ins()
                .fsub(
                    required_start,
                    current_storage_energy,
                );


        let additional_start =
            f64_max(
                &mut builder,
                zero_f64,
                additional_start_raw,
            );


        let pre_bucket_holding =
            builder.use_var(
                pre_bucket_holding_var,
            );


        let allocated_pv_total =
            builder.use_var(
                allocated_pv_total_var,
            );


        let unavoidable_deficit =
            builder.use_var(
                unavoidable_deficit_var,
            );


        let current_waterline =
            builder.use_var(
                current_waterline_var,
            );


        let current_hp_el =
            builder.use_var(
                current_hp_el_var,
            );


        let current_hp_th =
            builder
                .ins()
                .fmul(
                    current_hp_el,
                    cop,
                );


        store_output(
            &mut builder,
            output_pointer,
            OUT_KWH_NO_SUNSHINE,
            kwh_no_sunshine,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_BUCKET_DEMAND,
            bucket_demand_total,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_BUCKET_COOLING,
            bucket_cooling_total,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_PRE_BUCKET_HOLDING,
            pre_bucket_holding,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_ALLOCATED_PV,
            allocated_pv_total,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_UNAVOIDABLE_DEFICIT,
            unavoidable_deficit,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_REQUIRED_START,
            required_start,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_ADDITIONAL_START,
            additional_start,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CURRENT_WATERLINE,
            current_waterline,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CURRENT_HP_EL,
            current_hp_el,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CURRENT_HP_TH,
            current_hp_th,
        );


        builder
            .ins()
            .return_(
                &[],
            );


        builder.finalize(
            target_config,
        );
    }


    module
        .define_function(
            function_id,
            &mut context,
        )
        .map_err(
            |error| {
                format!(
                    "Could not define full no-sunshine \
                     JIT function: {error}"
                )
            },
        )?;


    module.clear_context(
        &mut context,
    );


    Ok(
        function_id,
    )
}



/*
 * ============================================================
 * CRANELIFT – 48 h CAUSAL SEARCH
 * ============================================================
 *
 * The parser selects this compiler by scope ID:
 *
 *   rust_peakShavingCausal
 *
 * The complete causal correction, including the 20-step binary
 * search, executes inside ONE JIT function.
 */


fn add_peak_shaving_causal_signature(
    module: &JITModule,
) -> cranelift::codegen::ir::Signature {

    let pointer_type =
        module
            .target_config()
            .pointer_type();


    let mut signature =
        module.make_signature();


    /*
     * Annual export and demand arrays.
     */
    signature.params.push(
        AbiParam::new(
            pointer_type,
        ),
    );

    signature.params.push(
        AbiParam::new(
            pointer_type,
        ),
    );


    /*
     * data_length
     * current_index
     * horizon_steps
     */
    for _ in 0..3 {
        signature.params.push(
            AbiParam::new(
                types::I64,
            ),
        );
    }


    /*
     * current_storage_energy
     * minimum_reserve_energy
     * target_storage_energy
     * maximum_heat_pump_electrical_power
     * cop
     * energy_cut_level
     * time_step_hours
     * storage_energy_per_kelvin
     * storage_conductance_w_per_kelvin
     * storage_min_temperature
     * ambient_temperature
     * daily_heat_loss
     */
    for _ in 0..12 {
        signature.params.push(
            AbiParam::new(
                types::F64,
            ),
        );
    }


    /*
     * Output array.
     */
    signature.params.push(
        AbiParam::new(
            pointer_type,
        ),
    );


    signature
}


fn build_peak_shaving_causal_function(
    module: &mut JITModule,
) -> Result<
    cranelift::module::FuncId,
    String,
> {

    let pointer_type =
        module
            .target_config()
            .pointer_type();


    if pointer_type != types::I64 {
        return Err(
            "The SunOpt causal JIT kernel requires a 64-bit host."
                .to_string(),
        );
    }


    let signature =
        add_peak_shaving_causal_signature(
            module,
        );


    let function_id =
        module
            .declare_function(
                "sunopt_peak_shaving_causal",
                Linkage::Local,
                &signature,
            )
            .map_err(
                |error| {
                    format!(
                        "Could not declare peak-shaving causal JIT function: {error}"
                    )
                },
            )?;


    let mut context =
        module.make_context();


    context.func.signature =
        signature;


    let target_config =
        module.target_config();


    let mut builder_context =
        FunctionBuilderContext::new();


    {
        let mut builder =
            FunctionBuilder::new(
                &mut context.func,
                &mut builder_context,
            );


        let entry =
            builder.create_block();

        let start_survival =
            builder.create_block();

        let survival_header =
            builder.create_block();

        let survival_body =
            builder.create_block();

        let current_bucket =
            builder.create_block();

        let future_bucket =
            builder.create_block();

        let reserve_check =
            builder.create_block();

        let future_charge =
            builder.create_block();

        let survival_failed =
            builder.create_block();

        let survival_succeeded =
            builder.create_block();

        let after_survival =
            builder.create_block();

        let phase_not_zero =
            builder.create_block();

        let phase_zero =
            builder.create_block();

        let phase_one =
            builder.create_block();

        let phase_two =
            builder.create_block();

        let normal_survives_finish =
            builder.create_block();

        let normal_failed =
            builder.create_block();

        let max_pv_failed =
            builder.create_block();

        let max_pv_survived_block =
            builder.create_block();

        let binary_continue =
            builder.create_block();

        let binary_finish =
            builder.create_block();

        let finish =
            builder.create_block();


        builder
            .append_block_params_for_function_params(
                entry,
            );


        builder.switch_to_block(
            entry,
        );


        let params =
            builder.block_params(
                entry,
            )
            .to_vec();


        let export_pointer =
            params[0];

        let demand_pointer =
            params[1];

        let data_length =
            params[2];

        let current_index =
            params[3];

        let horizon_steps =
            params[4];

        let current_storage_energy =
            params[5];

        let minimum_reserve_energy =
            params[6];

        let target_storage_energy =
            params[7];

        let maximum_hp_el =
            params[8];

        let cop =
            params[9];

        let energy_cut_level =
            params[10];

        let time_step_hours =
            params[11];

        let energy_per_kelvin =
            params[12];

        let conductance =
            params[13];

        let minimum_temperature =
            params[14];

        let ambient_temperature =
            params[15];

        let daily_heat_loss =
            params[16];

        let output_pointer =
            params[17];


        let zero_f64 =
            builder
                .ins()
                .f64const(
                    0.0,
                );

        let one_f64 =
            builder
                .ins()
                .f64const(
                    1.0,
                );

        let half_f64 =
            builder
                .ins()
                .f64const(
                    0.5,
                );

        let twenty_four =
            builder
                .ins()
                .f64const(
                    24.0,
                );

        let thousand =
            builder
                .ins()
                .f64const(
                    1000.0,
                );

        let epsilon =
            builder
                .ins()
                .f64const(
                    ENERGY_EPSILON,
                );


        let zero_i64 =
            builder
                .ins()
                .iconst(
                    types::I64,
                    0,
                );

        let one_i64 =
            builder
                .ins()
                .iconst(
                    types::I64,
                    1,
                );

        let two_i64 =
            builder
                .ins()
                .iconst(
                    types::I64,
                    2,
                );

        let twenty_i64 =
            builder
                .ins()
                .iconst(
                    types::I64,
                    20,
                );


        /*
         * State-machine variables.
         *
         * phase:
         *   0 = normal energy Waterline
         *   1 = Waterline zero
         *   2 = causal binary search
         */
        let phase_var =
            builder.declare_var(
                types::I64,
            );

        let candidate_cut_var =
            builder.declare_var(
                types::F64,
            );

        let simulated_energy_var =
            builder.declare_var(
                types::F64,
            );

        let offset_var =
            builder.declare_var(
                types::I64,
            );

        let survived_var =
            builder.declare_var(
                types::F64,
            );

        let lower_cut_var =
            builder.declare_var(
                types::F64,
            );

        let upper_cut_var =
            builder.declare_var(
                types::F64,
            );

        let binary_iteration_var =
            builder.declare_var(
                types::I64,
            );

        let binary_iteration_count_var =
            builder.declare_var(
                types::F64,
            );

        let survival_checks_var =
            builder.declare_var(
                types::F64,
            );

        let normal_survives_var =
            builder.declare_var(
                types::F64,
            );

        let maximum_pv_survives_var =
            builder.declare_var(
                types::F64,
            );

        let final_cut_var =
            builder.declare_var(
                types::F64,
            );


        let reserve_floor =
            f64_min(
                &mut builder,
                target_storage_energy,
                minimum_reserve_energy,
            );


        let current_non_negative =
            f64_max(
                &mut builder,
                zero_f64,
                current_storage_energy,
            );


        let clamped_current_storage =
            f64_min(
                &mut builder,
                target_storage_energy,
                current_non_negative,
            );


        builder.def_var(
            phase_var,
            zero_i64,
        );

        builder.def_var(
            candidate_cut_var,
            energy_cut_level,
        );

        builder.def_var(
            lower_cut_var,
            zero_f64,
        );

        builder.def_var(
            upper_cut_var,
            energy_cut_level,
        );

        builder.def_var(
            binary_iteration_var,
            zero_i64,
        );

        builder.def_var(
            binary_iteration_count_var,
            zero_f64,
        );

        builder.def_var(
            survival_checks_var,
            zero_f64,
        );

        builder.def_var(
            normal_survives_var,
            zero_f64,
        );

        builder.def_var(
            maximum_pv_survives_var,
            zero_f64,
        );

        builder.def_var(
            final_cut_var,
            energy_cut_level,
        );


        builder
            .ins()
            .jump(
                start_survival,
                &[],
            );


        /*
         * ----------------------------------------------------
         * START ONE survivesAtCutLevel(candidate)
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            start_survival,
        );


        let previous_checks =
            builder.use_var(
                survival_checks_var,
            );


        let next_checks =
            builder
                .ins()
                .fadd(
                    previous_checks,
                    one_f64,
                );


        builder.def_var(
            survival_checks_var,
            next_checks,
        );


        builder.def_var(
            simulated_energy_var,
            clamped_current_storage,
        );

        builder.def_var(
            offset_var,
            zero_i64,
        );

        builder.def_var(
            survived_var,
            one_f64,
        );


        builder
            .ins()
            .jump(
                survival_header,
                &[],
            );


        /*
         * ----------------------------------------------------
         * SURVIVAL LOOP HEADER
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            survival_header,
        );


        let offset =
            builder.use_var(
                offset_var,
            );


        let continue_survival =
            builder
                .ins()
                .icmp(
                    IntCC::SignedLessThan,
                    offset,
                    horizon_steps,
                );


        builder
            .ins()
            .brif(
                continue_survival,
                survival_body,
                &[],
                survival_succeeded,
                &[],
            );


        /*
         * ----------------------------------------------------
         * SURVIVAL LOOP BODY
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            survival_body,
        );


        let raw_index =
            builder
                .ins()
                .iadd(
                    current_index,
                    offset,
                );


        let data_index =
            builder
                .ins()
                .urem(
                    raw_index,
                    data_length,
                );


        let export_power =
            load_f64_indexed(
                &mut builder,
                export_pointer,
                data_index,
            );


        let candidate_cut =
            builder.use_var(
                candidate_cut_var,
            );


        let power_above_cut =
            builder
                .ins()
                .fsub(
                    export_power,
                    candidate_cut,
                );


        let positive_power_above_cut =
            f64_max(
                &mut builder,
                zero_f64,
                power_above_cut,
            );


        let hp_power =
            f64_min(
                &mut builder,
                maximum_hp_el,
                positive_power_above_cut,
            );


        let hp_times_cop =
            builder
                .ins()
                .fmul(
                    hp_power,
                    cop,
                );


        let hp_thermal_energy =
            builder
                .ins()
                .fmul(
                    hp_times_cop,
                    time_step_hours,
                );


        let is_current_bucket =
            builder
                .ins()
                .icmp(
                    IntCC::Equal,
                    offset,
                    zero_i64,
                );


        builder
            .ins()
            .brif(
                is_current_bucket,
                current_bucket,
                &[],
                future_bucket,
                &[],
            );


        /*
         * ----------------------------------------------------
         * OFFSET ZERO
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            current_bucket,
        );


        let current_simulated =
            builder.use_var(
                simulated_energy_var,
            );


        let current_plus_hp =
            builder
                .ins()
                .fadd(
                    current_simulated,
                    hp_thermal_energy,
                );


        let current_capped =
            f64_min(
                &mut builder,
                target_storage_energy,
                current_plus_hp,
            );


        builder.def_var(
            simulated_energy_var,
            current_capped,
        );


        let current_next_offset =
            builder
                .ins()
                .iadd(
                    offset,
                    one_i64,
                );

        builder.def_var(
            offset_var,
            current_next_offset,
        );

        builder
            .ins()
            .jump(
                survival_header,
                &[],
            );


        /*
         * ----------------------------------------------------
         * FUTURE BUCKET:
         * passive cooling -> demand -> reserve -> HP charge
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            future_bucket,
        );


        let simulated =
            builder.use_var(
                simulated_energy_var,
            );


        let daily_loss_times_dt =
            builder
                .ins()
                .fmul(
                    daily_heat_loss,
                    time_step_hours,
                );

        let fixed_loss =
            builder
                .ins()
                .fdiv(
                    daily_loss_times_dt,
                    twenty_four,
                );


        let min_minus_ambient =
            builder
                .ins()
                .fsub(
                    minimum_temperature,
                    ambient_temperature,
                );


        let energy_temperature_delta =
            builder
                .ins()
                .fdiv(
                    simulated,
                    energy_per_kelvin,
                );


        let temperature_delta =
            builder
                .ins()
                .fadd(
                    min_minus_ambient,
                    energy_temperature_delta,
                );


        let positive_temperature_delta =
            f64_max(
                &mut builder,
                zero_f64,
                temperature_delta,
            );


        let conductance_times_dt =
            builder
                .ins()
                .fmul(
                    conductance,
                    time_step_hours,
                );


        let conductance_loss_numerator =
            builder
                .ins()
                .fmul(
                    conductance_times_dt,
                    positive_temperature_delta,
                );


        let conductance_loss =
            builder
                .ins()
                .fdiv(
                    conductance_loss_numerator,
                    thousand,
                );


        let use_fixed_loss =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThanOrEqual,
                    daily_heat_loss,
                    zero_f64,
                );


        let requested_loss =
            builder
                .ins()
                .select(
                    use_fixed_loss,
                    fixed_loss,
                    conductance_loss,
                );


        let non_negative_loss =
            f64_max(
                &mut builder,
                zero_f64,
                requested_loss,
            );


        let heat_loss =
            f64_min(
                &mut builder,
                simulated,
                non_negative_loss,
            );


        let after_loss_raw =
            builder
                .ins()
                .fsub(
                    simulated,
                    heat_loss,
                );


        let after_loss =
            f64_max(
                &mut builder,
                zero_f64,
                after_loss_raw,
            );


        let demand_raw =
            load_f64_indexed(
                &mut builder,
                demand_pointer,
                data_index,
            );


        let demand =
            f64_max(
                &mut builder,
                zero_f64,
                demand_raw,
            );


        let discharged =
            f64_min(
                &mut builder,
                demand,
                after_loss,
            );


        let energy_before_forecast_charge =
            builder
                .ins()
                .fsub(
                    after_loss,
                    discharged,
                );


        let unmet_raw =
            builder
                .ins()
                .fsub(
                    demand,
                    discharged,
                );


        let unmet_demand =
            f64_max(
                &mut builder,
                zero_f64,
                unmet_raw,
            );


        let unmet_failure =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThan,
                    unmet_demand,
                    epsilon,
                );


        builder
            .ins()
            .brif(
                unmet_failure,
                survival_failed,
                &[],
                reserve_check,
                &[],
            );


        /*
         * ----------------------------------------------------
         * RESERVE CHECK
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            reserve_check,
        );


        let reserve_error =
            builder
                .ins()
                .fsub(
                    reserve_floor,
                    energy_before_forecast_charge,
                );


        let reserve_failure =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThan,
                    reserve_error,
                    epsilon,
                );


        builder
            .ins()
            .brif(
                reserve_failure,
                survival_failed,
                &[],
                future_charge,
                &[],
            );


        /*
         * Reserve survived: apply this future Waterline charge.
         */
        builder.switch_to_block(
            future_charge,
        );


        let future_plus_hp =
            builder
                .ins()
                .fadd(
                    energy_before_forecast_charge,
                    hp_thermal_energy,
                );


        let future_capped =
            f64_min(
                &mut builder,
                target_storage_energy,
                future_plus_hp,
            );


        builder.def_var(
            simulated_energy_var,
            future_capped,
        );


        let future_next_offset =
            builder
                .ins()
                .iadd(
                    offset,
                    one_i64,
                );


        builder.def_var(
            offset_var,
            future_next_offset,
        );


        builder
            .ins()
            .jump(
                survival_header,
                &[],
            );


        /*
         * ----------------------------------------------------
         * EARLY FAILURE
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            survival_failed,
        );


        builder.def_var(
            survived_var,
            zero_f64,
        );


        builder
            .ins()
            .jump(
                after_survival,
                &[],
            );


        /*
         * ----------------------------------------------------
         * COMPLETE SUCCESS
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            survival_succeeded,
        );


        builder.def_var(
            survived_var,
            one_f64,
        );


        builder
            .ins()
            .jump(
                after_survival,
                &[],
            );


        /*
         * ----------------------------------------------------
         * DISPATCH RESULT BY SEARCH PHASE
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            after_survival,
        );


        let phase =
            builder.use_var(
                phase_var,
            );


        let phase_is_zero =
            builder
                .ins()
                .icmp(
                    IntCC::Equal,
                    phase,
                    zero_i64,
                );


        builder
            .ins()
            .brif(
                phase_is_zero,
                phase_zero,
                &[],
                phase_not_zero,
                &[],
            );


        builder.switch_to_block(
            phase_not_zero,
        );


        let phase_is_one =
            builder
                .ins()
                .icmp(
                    IntCC::Equal,
                    phase,
                    one_i64,
                );


        builder
            .ins()
            .brif(
                phase_is_one,
                phase_one,
                &[],
                phase_two,
                &[],
            );


        /*
         * ----------------------------------------------------
         * PHASE 0: NORMAL ENERGY WATERLINE
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            phase_zero,
        );


        let normal_survived =
            builder.use_var(
                survived_var,
            );


        builder.def_var(
            normal_survives_var,
            normal_survived,
        );


        let normal_is_true =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThan,
                    normal_survived,
                    half_f64,
                );


        builder
            .ins()
            .brif(
                normal_is_true,
                normal_survives_finish,
                &[],
                normal_failed,
                &[],
            );


        builder.switch_to_block(
            normal_survives_finish,
        );


        builder.def_var(
            maximum_pv_survives_var,
            one_f64,
        );

        builder.def_var(
            final_cut_var,
            energy_cut_level,
        );


        builder
            .ins()
            .jump(
                finish,
                &[],
            );


        builder.switch_to_block(
            normal_failed,
        );


        builder.def_var(
            phase_var,
            one_i64,
        );

        builder.def_var(
            candidate_cut_var,
            zero_f64,
        );


        builder
            .ins()
            .jump(
                start_survival,
                &[],
            );


        /*
         * ----------------------------------------------------
         * PHASE 1: WATERLINE ZERO
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            phase_one,
        );


        let max_pv_survived_value =
            builder.use_var(
                survived_var,
            );


        builder.def_var(
            maximum_pv_survives_var,
            max_pv_survived_value,
        );


        let max_pv_is_true =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThan,
                    max_pv_survived_value,
                    half_f64,
                );


        builder
            .ins()
            .brif(
                max_pv_is_true,
                max_pv_survived_block,
                &[],
                max_pv_failed,
                &[],
            );


        builder.switch_to_block(
            max_pv_failed,
        );


        builder.def_var(
            final_cut_var,
            zero_f64,
        );


        builder
            .ins()
            .jump(
                finish,
                &[],
            );


        builder.switch_to_block(
            max_pv_survived_block,
        );


        builder.def_var(
            phase_var,
            two_i64,
        );

        builder.def_var(
            lower_cut_var,
            zero_f64,
        );

        builder.def_var(
            upper_cut_var,
            energy_cut_level,
        );

        builder.def_var(
            binary_iteration_var,
            zero_i64,
        );


        let first_middle =
            builder
                .ins()
                .fmul(
                    energy_cut_level,
                    half_f64,
                );


        builder.def_var(
            candidate_cut_var,
            first_middle,
        );


        builder
            .ins()
            .jump(
                start_survival,
                &[],
            );


        /*
         * ----------------------------------------------------
         * PHASE 2: BINARY SEARCH RESULT
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            phase_two,
        );


        let middle_survived =
            builder.use_var(
                survived_var,
            );


        let current_candidate =
            builder.use_var(
                candidate_cut_var,
            );


        let middle_is_true =
            builder
                .ins()
                .fcmp(
                    FloatCC::GreaterThan,
                    middle_survived,
                    half_f64,
                );


        let old_lower =
            builder.use_var(
                lower_cut_var,
            );

        let old_upper =
            builder.use_var(
                upper_cut_var,
            );


        let next_lower =
            builder
                .ins()
                .select(
                    middle_is_true,
                    current_candidate,
                    old_lower,
                );


        let next_upper =
            builder
                .ins()
                .select(
                    middle_is_true,
                    old_upper,
                    current_candidate,
                );


        builder.def_var(
            lower_cut_var,
            next_lower,
        );

        builder.def_var(
            upper_cut_var,
            next_upper,
        );


        let previous_binary_iteration =
            builder.use_var(
                binary_iteration_var,
            );


        let next_binary_iteration =
            builder
                .ins()
                .iadd(
                    previous_binary_iteration,
                    one_i64,
                );


        builder.def_var(
            binary_iteration_var,
            next_binary_iteration,
        );


        let previous_binary_count =
            builder.use_var(
                binary_iteration_count_var,
            );


        let next_binary_count =
            builder
                .ins()
                .fadd(
                    previous_binary_count,
                    one_f64,
                );


        builder.def_var(
            binary_iteration_count_var,
            next_binary_count,
        );


        let continue_binary =
            builder
                .ins()
                .icmp(
                    IntCC::SignedLessThan,
                    next_binary_iteration,
                    twenty_i64,
                );


        builder
            .ins()
            .brif(
                continue_binary,
                binary_continue,
                &[],
                binary_finish,
                &[],
            );


        builder.switch_to_block(
            binary_continue,
        );


        let lower_for_middle =
            builder.use_var(
                lower_cut_var,
            );

        let upper_for_middle =
            builder.use_var(
                upper_cut_var,
            );


        let lower_plus_upper =
            builder
                .ins()
                .fadd(
                    lower_for_middle,
                    upper_for_middle,
                );


        let next_middle =
            builder
                .ins()
                .fmul(
                    lower_plus_upper,
                    half_f64,
                );


        builder.def_var(
            candidate_cut_var,
            next_middle,
        );


        builder
            .ins()
            .jump(
                start_survival,
                &[],
            );


        builder.switch_to_block(
            binary_finish,
        );


        let final_lower =
            builder.use_var(
                lower_cut_var,
            );


        builder.def_var(
            final_cut_var,
            final_lower,
        );


        builder
            .ins()
            .jump(
                finish,
                &[],
            );


        /*
         * ----------------------------------------------------
         * OUTPUT
         * ----------------------------------------------------
         */

        builder.switch_to_block(
            finish,
        );


        let output_cut =
            builder.use_var(
                final_cut_var,
            );

        let output_normal_survives =
            builder.use_var(
                normal_survives_var,
            );

        let output_max_pv_survives =
            builder.use_var(
                maximum_pv_survives_var,
            );

        let output_survival_checks =
            builder.use_var(
                survival_checks_var,
            );

        let output_binary_iterations =
            builder.use_var(
                binary_iteration_count_var,
            );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CAUSAL_CUT_LEVEL,
            output_cut,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CAUSAL_NORMAL_SURVIVES,
            output_normal_survives,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CAUSAL_MAX_PV_SURVIVES,
            output_max_pv_survives,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CAUSAL_SURVIVAL_CHECKS,
            output_survival_checks,
        );


        store_output(
            &mut builder,
            output_pointer,
            OUT_CAUSAL_BINARY_ITERATIONS,
            output_binary_iterations,
        );


        builder
            .ins()
            .return_(
                &[],
            );


        /*
         * All predecessors are known now.
         */
        builder.seal_all_blocks();


        builder.finalize(
            target_config,
        );
    }


    module
        .define_function(
            function_id,
            &mut context,
        )
        .map_err(
            |error| {
                format!(
                    "Could not define peak-shaving causal JIT function: {error}"
                )
            },
        )?;


    module.clear_context(
        &mut context,
    );


    Ok(
        function_id,
    )
}


/*
 * ============================================================
 * PUBLIC COMPILER
 * ============================================================
 */

pub fn compile_no_sunshine_kernel_from_typescript(
    source: &str,
    file_path: &Path,
) -> Result<
    CompiledNoSunshineKernel,
    String,
> {

    let source_hash =
        validate_scope_source(
            source,
            file_path,
            "rust_noSunshine",
        )?;


    let jit_builder =
        JITBuilder::new(
            default_libcall_names(),
        )
        .map_err(
            |error| {
                format!(
                    "Could not create Cranelift JIT \
                     builder: {error}"
                )
            },
        )?;


    let mut module =
        JITModule::new(
            jit_builder,
        );


    let function_id =
        build_no_sunshine_function(
            &mut module,
        )?;


    module
        .finalize_definitions()
        .map_err(
            |error| {
                format!(
                    "Could not finalise full \
                     no-sunshine JIT function: {error}"
                )
            },
        )?;


    let code_pointer =
        module.get_finalized_function(
            function_id,
        );


    Ok(
        CompiledNoSunshineKernel {
            _module:
                module,

            code_pointer,

            source_hash,
        },
    )
}


pub fn compile_peak_shaving_causal_kernel_from_typescript(
    source: &str,
    file_path: &Path,
) -> Result<
    CompiledPeakShavingCausalKernel,
    String,
> {

    let source_hash =
        validate_scope_source(
            source,
            file_path,
            "rust_peakShavingCausal",
        )?;


    let jit_builder =
        JITBuilder::new(
            default_libcall_names(),
        )
        .map_err(
            |error| {
                format!(
                    "Could not create Cranelift JIT builder: {error}"
                )
            },
        )?;


    let mut module =
        JITModule::new(
            jit_builder,
        );


    let function_id =
        build_peak_shaving_causal_function(
            &mut module,
        )?;


    module
        .finalize_definitions()
        .map_err(
            |error| {
                format!(
                    "Could not finalise peak-shaving causal JIT function: {error}"
                )
            },
        )?;


    let code_pointer =
        module.get_finalized_function(
            function_id,
        );


    Ok(
        CompiledPeakShavingCausalKernel {
            _module:
                module,

            code_pointer,

            source_hash,
        },
    )
}


/*
 * One global compiler entry point.
 *
 * The parser source remains the same TypeScript file; scope_id
 * chooses the rust_* block inside that source.
 */
pub fn compile_acceleration_scope_from_typescript(
    source: &str,
    file_path: &Path,
    scope_id: &str,
) -> Result<
    CompiledAccelerationScope,
    String,
> {

    match scope_id {

        "rust_noSunshine" =>
            Ok(
                CompiledAccelerationScope::NoSunshine(
                    compile_no_sunshine_kernel_from_typescript(
                        source,
                        file_path,
                    )?,
                ),
            ),

        "rust_peakShavingCausal" =>
            Ok(
                CompiledAccelerationScope::PeakShavingCausal(
                    compile_peak_shaving_causal_kernel_from_typescript(
                        source,
                        file_path,
                    )?,
                ),
            ),

        other =>
            Err(
                format!(
                    "Unsupported SunOpt acceleration scope: {other}"
                ),
            ),
    }
}


/*
 * ============================================================
 * TESTS
 * ============================================================
 */

#[cfg(test)]
mod tests {

    use super::*;


    const TEST_SOURCE: &str = r#"
        export function test() {

            rust_noSunshine: {

                for (
                    let offset = horizonSteps - 1;
                    offset >= 0;
                    offset--
                ) {
                    const exportPower =
                        Math.max(
                            0,
                            weatherPoint.pvPower -
                                householdReservePower,
                        );

                    requiredEnergyAfterStep =
                        requiredEnergyBeforeStep;
                }
            }
        }
    "#;


    fn test_parameters()
        -> NoSunshineKernelParameters {

        NoSunshineKernelParameters {
            current_index: 173,

            horizon_steps: 672,

            protected_steps: 192,

            time_step_hours: 0.25,

            household_reserve_power: 2.0,

            maximum_heat_pump_electrical_power:
                5.0 / 3.0,

            cop: 3.0,

            initial_waterline_power: 10.0,

            target_storage_energy: 52.325,

            storage_energy_per_kelvin:
                3000.0 *
                4.186 /
                3600.0,

            storage_conductance_w_per_kelvin:
                3.617,

            storage_min_temperature: 60.0,

            ambient_temperature: 20.0,

            daily_heat_loss: -1.0,

            current_storage_energy: 10.0,
        }
    }


    fn test_series()
        -> (
            Vec<f64>,
            Vec<f64>,
        ) {

        let length =
            1024;


        let mut pv =
            Vec::with_capacity(
                length,
            );


        let mut load =
            Vec::with_capacity(
                length,
            );


        for index in 0..length {

            let quarter =
                index % 96;


            let hour =
                quarter as f64 /
                4.0;


            let pv_power =
                if (
                    6.0..=18.0
                ).contains(
                    &hour,
                ) {

                    let phase =
                        (
                            hour -
                            6.0
                        ) /
                        12.0 *
                        std::f64::consts::PI;


                    8.5 *
                    phase.sin()
                        .max(
                            0.0,
                        )

                } else {
                    0.0
                };


            let demand =
                if quarter == 28 {
                    0.35
                } else if quarter == 80 {
                    0.28
                } else {
                    0.0
                };


            pv.push(
                pv_power,
            );

            load.push(
                demand,
            );
        }


        (
            pv,
            load,
        )
    }


    fn assert_close(
        name: &str,
        left: f64,
        right: f64,
    ) {

        let tolerance =
            1e-9;


        assert!(
            (
                left -
                right
            ).abs() <
            tolerance,

            "{name}: left={left}, right={right}",
        );
    }


    #[test]
    fn oxc_finds_full_no_sunshine_payload() {

        let scopes =
            inspect_rust_payloads(
                TEST_SOURCE,
                Path::new(
                    "test.ts",
                ),
            )
            .expect(
                "TypeScript should parse.",
            );


        assert_eq!(
            scopes.len(),
            1,
        );


        assert_eq!(
            scopes[0].label,
            "rust_noSunshine",
        );


        assert_eq!(
            scopes[0]
                .first_statement_kind
                .as_deref(),
            Some(
                "ForStatement",
            ),
        );
    }


    #[test]
    fn full_jit_matches_reference_for_672_buckets() {

        let (
            pv,
            load,
        ) =
            test_series();


        let parameters =
            test_parameters();


        let reference =
            run_no_sunshine_reference(
                &pv,
                &load,
                parameters,
            )
            .expect(
                "Reference simulation should run.",
            );


        let kernel =
            compile_no_sunshine_kernel_from_typescript(
                TEST_SOURCE,
                Path::new(
                    "test.ts",
                ),
            )
            .expect(
                "JIT kernel should compile.",
            );


        let jit =
            kernel
                .execute(
                    &pv,
                    &load,
                    parameters,
                )
                .expect(
                    "JIT simulation should run.",
                );


        println!(
            "source hash: {}",
            kernel.source_hash(),
        );

        println!(
            "reference: {reference:#?}"
        );

        println!(
            "jit: {jit:#?}"
        );


        assert_close(
            "kwh_no_sunshine",
            reference.kwh_no_sunshine,
            jit.kwh_no_sunshine,
        );

        assert_close(
            "bucket_demand_energy",
            reference.bucket_demand_energy,
            jit.bucket_demand_energy,
        );

        assert_close(
            "bucket_cooling_loss_energy",
            reference.bucket_cooling_loss_energy,
            jit.bucket_cooling_loss_energy,
        );

        assert_close(
            "pre_bucket_holding_loss_energy",
            reference.pre_bucket_holding_loss_energy,
            jit.pre_bucket_holding_loss_energy,
        );

        assert_close(
            "allocated_pv_thermal_energy",
            reference.allocated_pv_thermal_energy,
            jit.allocated_pv_thermal_energy,
        );

        assert_close(
            "unavoidable_thermal_deficit",
            reference.unavoidable_thermal_deficit,
            jit.unavoidable_thermal_deficit,
        );

        assert_close(
            "required_storage_energy_at_start",
            reference.required_storage_energy_at_start,
            jit.required_storage_energy_at_start,
        );

        assert_close(
            "additional_storage_energy_needed_at_start",
            reference.additional_storage_energy_needed_at_start,
            jit.additional_storage_energy_needed_at_start,
        );

        assert_close(
            "current_waterline_power",
            reference.current_waterline_power,
            jit.current_waterline_power,
        );

        assert_close(
            "current_scheduled_hp_electrical_power",
            reference.current_scheduled_hp_electrical_power,
            jit.current_scheduled_hp_electrical_power,
        );

        assert_close(
            "current_scheduled_hp_thermal_power",
            reference.current_scheduled_hp_thermal_power,
            jit.current_scheduled_hp_thermal_power,
        );
    }


    #[test]
    fn source_change_changes_kernel_hash() {

        let first =
            compile_no_sunshine_kernel_from_typescript(
                TEST_SOURCE,
                Path::new(
                    "test.ts",
                ),
            )
            .expect(
                "First kernel should compile.",
            );


        let changed_source =
            TEST_SOURCE.replace(
                "householdReservePower",
                "householdReservePower + 0",
            );


        let second =
            compile_no_sunshine_kernel_from_typescript(
                &changed_source,
                Path::new(
                    "test.ts",
                ),
            )
            .expect(
                "Second kernel should compile.",
            );


        assert_ne!(
            first.source_hash(),
            second.source_hash(),
        );
    }
}


/*
 * ============================================================
 * GLOBAL SCOPE ARCHITECTURE TESTS
 * ============================================================
 */

#[cfg(test)]
mod global_scope_architecture_tests {

    use super::*;


    const TWO_SCOPE_SOURCE:
        &str =
        r#"
            export function test() {

                rust_noSunshine: {
                    for (
                        let offset = 10;
                        offset >= 0;
                        offset--
                    ) {
                        const exportPower = 1;
                    }
                }

                rust_peakShavingCausal: {
                    let cut = 1;

                    for (
                        let iteration = 0;
                        iteration < 20;
                        iteration++
                    ) {
                        cut = cut / 2;
                    }
                }
            }
        "#;


    #[test]
    fn oxc_discovers_multiple_acceleration_scopes_in_one_source() {

        let scopes =
            inspect_rust_payloads(
                TWO_SCOPE_SOURCE,
                Path::new(
                    "rust-acceleration-scopes.ts",
                ),
            )
            .expect(
                "Global acceleration TypeScript should parse.",
            );


        let labels =
            scopes
                .iter()
                .map(
                    |scope| {
                        scope.label.as_str()
                    },
                )
                .collect::<Vec<_>>();


        assert!(
            labels.contains(
                &"rust_noSunshine",
            ),
        );


        assert!(
            labels.contains(
                &"rust_peakShavingCausal",
            ),
        );
    }


    #[test]
    fn scope_id_is_part_of_cache_identity() {

        let no_sunshine =
            acceleration_scope_source_hash(
                TWO_SCOPE_SOURCE,
                Path::new(
                    "rust-acceleration-scopes.ts",
                ),
                "rust_noSunshine",
            )
            .expect(
                "No-sunshine scope should hash.",
            );


        let causal =
            acceleration_scope_source_hash(
                TWO_SCOPE_SOURCE,
                Path::new(
                    "rust-acceleration-scopes.ts",
                ),
                "rust_peakShavingCausal",
            )
            .expect(
                "Causal scope should hash.",
            );


        assert_ne!(
            no_sunshine,
            causal,
        );
    }
}
