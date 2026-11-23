/*
 * Ansatz A — generic TypeScript numeric source -> C++/CUDA compiler.
 * NO application-specific operation, scope ID, physical constant or kernel.
 * Experimental standalone research module. It does NOT alter production.
 * Supported: pure scalar functions, up to eight number arguments, numeric
 * arithmetic/comparisons, conditionals, bounded for loops, Math operations.
 * Deliberately rejects unsupported syntax instead of falling back silently.
 */
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import path from 'node:path';

import { loadProjectTypeScript } from './approach-a-generic-typescript.mjs';
const ts = loadProjectTypeScript(import.meta.url);
const fail = (node, message, sourceFile) => {
  const line = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
  throw new Error(`UNSUPPORTED ${sourceFile.fileName}:${line.line + 1}:${line.character + 1}: ${message}`);
};
const str = (s) => JSON.stringify(s);
const ident = (name) => /^[A-Za-z_$][\w$]*$/.test(name) && !/^(auto|double|for|return|if|else|int|while|float|void|namespace|class|template|new)$/.test(name);
const litNum = (text) => /^\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$/.test(text) || /^\.\d+(?:[eE][-+]?\d+)?$/.test(text);
const ops = new Map([
  [ts.SyntaxKind.PlusToken, '+'], [ts.SyntaxKind.MinusToken, '-'],
  [ts.SyntaxKind.AsteriskToken, '*'], [ts.SyntaxKind.SlashToken, '/'],
  [ts.SyntaxKind.PercentToken, '%'],
  [ts.SyntaxKind.LessThanToken, '<'], [ts.SyntaxKind.LessThanEqualsToken, '<='],
  [ts.SyntaxKind.GreaterThanToken, '>'], [ts.SyntaxKind.GreaterThanEqualsToken, '>='],
  [ts.SyntaxKind.EqualsEqualsEqualsToken, '=='], [ts.SyntaxKind.ExclamationEqualsEqualsToken, '!='],
  [ts.SyntaxKind.EqualsEqualsToken, '=='], [ts.SyntaxKind.ExclamationEqualsToken, '!='],
  [ts.SyntaxKind.AmpersandAmpersandToken, '&&'], [ts.SyntaxKind.BarBarToken, '||'],
]);
const compound = new Map([
  [ts.SyntaxKind.EqualsToken, '='], [ts.SyntaxKind.PlusEqualsToken, '+='],
  [ts.SyntaxKind.MinusEqualsToken, '-='], [ts.SyntaxKind.AsteriskEqualsToken, '*='],
  [ts.SyntaxKind.SlashEqualsToken, '/='],
]);
const mathFn = new Map([
  ['min','fmin'], ['max','fmax'], ['abs','fabs'], ['sqrt','sqrt'],
  ['floor','floor'], ['ceil','ceil'], ['round','round'], ['trunc','trunc'],
  ['exp','exp'], ['log','log'], ['log2','log2'], ['sin','sin'], ['cos','cos'],
  ['tan','tan'], ['pow','pow'], ['atan2','atan2'],
]);

export function compileGenericNumericFunction(source, requestedName) {
  const file = ts.createSourceFile('generic-numeric.ts', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  if (file.parseDiagnostics.length) throw new Error('TS syntax error: ' + ts.flattenDiagnosticMessageText(file.parseDiagnostics[0].messageText, '\n'));
  const functions = file.statements.filter(ts.isFunctionDeclaration);
  if (functions.length !== 1 || file.statements.length !== 1)
    throw new Error('Require a standalone function declaration; no imports, globals or other statements.');
  const fn = functions[0];
  if (!fn.name || !fn.body) fail(fn, 'named function with body required', file);
  if (requestedName && fn.name.text !== requestedName) fail(fn, `function ${requestedName} was not supplied`, file);
  const params = fn.parameters.map(p => {
    if (!ts.isIdentifier(p.name) || p.questionToken || p.dotDotDotToken || p.initializer || !p.type || p.type.kind !== ts.SyntaxKind.NumberKeyword)
      fail(p, 'only required scalar parameters of type number', file);
    if (!ident(p.name.text)) fail(p, 'unsupported parameter identifier', file);
    return p.name.text;
  });
  if (params.length < 1 || params.length > 8 || new Set(params).size !== params.length) fail(fn, 'one to eight distinct numeric arguments required', file);
  if (!fn.type || fn.type.kind !== ts.SyntaxKind.NumberKeyword) fail(fn, 'explicit number return type required', file);
  const seen = new Set(params);
  const scope = () => new Set(seen);
  function compileExpr(expr, names) {
    if (ts.isParenthesizedExpression(expr)) return `(${compileExpr(expr.expression, names)})`;
    if (ts.isNumericLiteral(expr)) { if (!litNum(expr.text)) fail(expr, 'invalid numeric literal', file); return Number(expr.text).toString() + '.0'.replace(/^.*$/, Number.isInteger(Number(expr.text)) ? '.0' : ''); }
    if (ts.isIdentifier(expr)) {
      if (!names.has(expr.text)) fail(expr, `unknown variable ${expr.text}`, file);
      return expr.text;
    }
    if (ts.isPrefixUnaryExpression(expr)) {
      const t = expr.operator;
      if ([ts.SyntaxKind.PlusToken, ts.SyntaxKind.MinusToken, ts.SyntaxKind.ExclamationToken].includes(t))
        return `(${ts.tokenToString(t)}${compileExpr(expr.operand,names)})`;
    }
    if (ts.isBinaryExpression(expr)) {
      if (expr.operatorToken.kind === ts.SyntaxKind.AsteriskAsteriskToken)
        return `pow(${compileExpr(expr.left,names)}, ${compileExpr(expr.right,names)})`;
      const op = ops.get(expr.operatorToken.kind);
      if (op) {
        if (op === '%') return `fmod(${compileExpr(expr.left,names)}, ${compileExpr(expr.right,names)})`;
        return `(${compileExpr(expr.left,names)} ${op} ${compileExpr(expr.right,names)})`;
      }
    }
    if (ts.isConditionalExpression(expr))
      return `(${compileExpr(expr.condition,names)} ? ${compileExpr(expr.whenTrue,names)} : ${compileExpr(expr.whenFalse,names)})`;
    if (ts.isCallExpression(expr) && ts.isPropertyAccessExpression(expr.expression) &&
        ts.isIdentifier(expr.expression.expression) && expr.expression.expression.text === 'Math') {
      const name = expr.expression.name.text;
      if (!mathFn.has(name)) fail(expr, `Math.${name} is not supported`, file);
      if (name === 'min' || name === 'max') {
        if (expr.arguments.length < 2) fail(expr, 'min/max need two or more arguments', file);
        return expr.arguments.slice(1).reduce((acc,a) => `${mathFn.get(name)}(${acc}, ${compileExpr(a,names)})`,compileExpr(expr.arguments[0],names));
      }
      const arity = ['pow','atan2'].includes(name) ? 2 : 1;
      if (expr.arguments.length !== arity) fail(expr, `Math.${name} requires ${arity} arguments`, file);
      return `${mathFn.get(name)}(${expr.arguments.map(a=>compileExpr(a,names)).join(', ')})`;
    }
    fail(expr, `expression ${ts.SyntaxKind[expr.kind]} is not supported`, file);
  }
  function compileStatement(stmt, names, depth=0) {
    if (depth>40) fail(stmt, 'nested statement limit exceeded', file);
    if (ts.isBlock(stmt)) {
      const blockScope = new Set(names);
      const code = stmt.statements.map(s => compileStatement(s,blockScope,depth+1)).join('\n');
      return `{\n${code}\n}`;
    }
    if (ts.isVariableStatement(stmt)) {
      if (stmt.declarationList.declarations.length < 1) fail(stmt, 'empty declaration',file);
      return stmt.declarationList.declarations.map(d=> {
        if (!ts.isIdentifier(d.name) || !d.initializer || !ident(d.name.text) || names.has(d.name.text)) fail(d,'numeric let with unique name and initializer required',file);
        const value=compileExpr(d.initializer,names);
        names.add(d.name.text);
        return `double ${d.name.text} = ${value};`;
      }).join('\n');
    }
    if (ts.isReturnStatement(stmt)) {
      if (!stmt.expression) fail(stmt,'numeric return value required',file);
      return `return ${compileExpr(stmt.expression,names)};`;
    }
    if (ts.isIfStatement(stmt)) {
      return `if (${compileExpr(stmt.expression,names)}) ${compileStatement(stmt.thenStatement,new Set(names),depth+1)}${stmt.elseStatement ? ` else ${compileStatement(stmt.elseStatement,new Set(names),depth+1)}` : ''}`;
    }
    if (ts.isExpressionStatement(stmt) && ts.isBinaryExpression(stmt.expression)) {
      const v=stmt.expression;
      if (!ts.isIdentifier(v.left) || !names.has(v.left.text) || !compound.has(v.operatorToken.kind)) fail(v,'assignment to declared numeric variable only',file);
      return `${v.left.text} ${compound.get(v.operatorToken.kind)} ${compileExpr(v.right,names)};`;
    }
    if (ts.isForStatement(stmt)) {
      if (!stmt.initializer || !ts.isVariableDeclarationList(stmt.initializer) || stmt.initializer.declarations.length !== 1) fail(stmt,'bounded for loop requires a single numeric index declaration',file);
      const d=stmt.initializer.declarations[0];
      if (!ts.isIdentifier(d.name) || !ident(d.name.text) || !d.initializer || !ts.isNumericLiteral(d.initializer)) fail(stmt,'for index must start at a numeric literal',file);
      if (names.has(d.name.text)) fail(stmt,'duplicate index',file);
      if (!stmt.condition || !ts.isBinaryExpression(stmt.condition) || !ts.isIdentifier(stmt.condition.left) || stmt.condition.left.text!==d.name.text || !ts.isNumericLiteral(stmt.condition.right)) fail(stmt,'for bound must be a numeric literal comparison of its index',file);
      const cmp = ops.get(stmt.condition.operatorToken.kind);
      if (!['<','<=','>','>='].includes(cmp)) fail(stmt,'invalid for comparison',file);
      if (!stmt.incrementor || (!ts.isPostfixUnaryExpression(stmt.incrementor) && !ts.isPrefixUnaryExpression(stmt.incrementor))) fail(stmt,'for increment must be ++ or --',file);
      const incr=stmt.incrementor;
      if (!ts.isIdentifier(incr.operand) || incr.operand.text!==d.name.text || ![ts.SyntaxKind.PlusPlusToken,ts.SyntaxKind.MinusMinusToken].includes(incr.operator)) fail(stmt,'for index may only use ++ or --',file);
      const start=Number(d.initializer.text),stop=Number(stmt.condition.right.text);
      const up=incr.operator===ts.SyntaxKind.PlusPlusToken;
      if ((up && !['<','<='].includes(cmp)) || (!up && !['>','>='].includes(cmp))) fail(stmt,'for loop bound does not converge',file);
      const count=up ? stop-start+(cmp==='<='?1:0) : start-stop+(cmp==='>='?1:0);
      if (count<0 || count>16384 || !Number.isInteger(count)) fail(stmt,'for loop bound must be finite (<=16384)',file);
      const next=new Set(names); next.add(d.name.text);
      const body=compileStatement(stmt.statement,next,depth+1);
      return `for (int ${d.name.text} = ${start}; ${d.name.text} ${cmp} ${stop}; ${d.name.text}${up?'++':'--'}) ${body}`;
    }
    fail(stmt,`statement ${ts.SyntaxKind[stmt.kind]} is not supported`,file);
  }
  const body=compileStatement(fn.body,scope());
  const cpp=`double ${fn.name.text}(${params.map(p=>`double ${p}`).join(', ')}) ${body}`;
  const digest=createHash('sha256').update(source).digest('hex');
  // This is an emitted program, not a prewritten algorithm for any function.
  return Object.freeze({ name:fn.name.text, params, arity:params.length, body:cpp, hash:digest });
}

export function makeCppHost(compiled, rows) {
  const values=rows.flat().map(n=>Number(n).toString() + (Number.isInteger(Number(n)) ? '.0' : '')).join(',');
  return `#include <cmath>\n#include <iostream>\n#include <iomanip>\n${compiled.body}\nint main(){\n static const double data[]={${values}};\n std::cout << std::setprecision(17);\n for(size_t i=0;i<${rows.length};i++){std::cout<<${compiled.name}(${compiled.params.map((_,i)=>`data[i*${compiled.arity}+${i}]`).join(',')})<<"\\n";}\n}\n`;
}

export function makeCudaHost(compiled, rows) {
  const values=rows.flat().map(n=>Number(n).toString() + (Number.isInteger(Number(n)) ? '.0' : '')).join(',');
  return `#include <cuda_runtime.h>\n#include <cmath>\n#include <iostream>\n#include <iomanip>\n#include <vector>\n#include <chrono>\n#define CUDA_CHECK(c) do{auto e=(c);if(e!=cudaSuccess){std::cerr<<"CUDA_ERROR "<<cudaGetErrorString(e)<<"\\n";return 2;}}while(0)\n__device__ __forceinline__ ${compiled.body}\n__global__ void generic_kernel(const double* input,double* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=${compiled.name}(${compiled.params.map((_,i)=>`input[i*${compiled.arity}+${i}]`).join(',')});}\nint main(){\nconstexpr int N=${rows.length};\nstatic const double data[]={${values}};\nstd::vector<double> output(N);double *dIn=nullptr,*dOut=nullptr;\nCUDA_CHECK(cudaMalloc((void**)&dIn,sizeof(data)));CUDA_CHECK(cudaMalloc((void**)&dOut,N*sizeof(double)));\nCUDA_CHECK(cudaMemcpy(dIn,data,sizeof(data),cudaMemcpyHostToDevice));\nfor(int j=0;j<3;j++){generic_kernel<<<(N+255)/256,256>>>(dIn,dOut,N);CUDA_CHECK(cudaDeviceSynchronize());}\nauto t0=std::chrono::steady_clock::now();\nfor(int j=0;j<100;j++){generic_kernel<<<(N+255)/256,256>>>(dIn,dOut,N);}\nCUDA_CHECK(cudaDeviceSynchronize());\nauto t1=std::chrono::steady_clock::now();\nCUDA_CHECK(cudaMemcpy(output.data(),dOut,N*sizeof(double),cudaMemcpyDeviceToHost));\nstd::cout<<std::setprecision(17);for(double v:output)std::cout<<v<<"\\n";\nstd::cerr<<"KERNEL_100_MS "<<std::chrono::duration<double,std::milli>(t1-t0).count()<<"\\n";\nCUDA_CHECK(cudaFree(dIn));CUDA_CHECK(cudaFree(dOut));\n}\n`;
}
