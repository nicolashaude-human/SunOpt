/*
 * SunOpt research / Approach A: structural TypeScript -> C++/CUDA lowering.
 * No application function, domain equation, scope ID or physical constant
 * is embedded in this translator. On unsupported syntax, reject, never fall
 * back to an opaque pre-written solver.
 */
import { createHash } from 'node:crypto';
import { loadProjectTypeScript } from './approach-a-generic-typescript.mjs';
const ts = loadProjectTypeScript(import.meta.url);
const key = createHash;
const forbidden = new Set(['auto','double','float','void','return','int','long','short','for','if','else','while','class','struct','const','new','delete','namespace','template','true','false']);
const safe = (name) => /^[A-Za-z_$][\w$]*$/.test(name) && !forbidden.has(name);
const opMap = new Map([
  [ts.SyntaxKind.PlusToken,'+'],[ts.SyntaxKind.MinusToken,'-'],[ts.SyntaxKind.AsteriskToken,'*'],[ts.SyntaxKind.SlashToken,'/'],
  [ts.SyntaxKind.LessThanToken,'<'],[ts.SyntaxKind.LessThanEqualsToken,'<='],[ts.SyntaxKind.GreaterThanToken,'>'],
  [ts.SyntaxKind.GreaterThanEqualsToken,'>='],[ts.SyntaxKind.EqualsEqualsEqualsToken,'=='],[ts.SyntaxKind.ExclamationEqualsEqualsToken,'!='],
  [ts.SyntaxKind.EqualsEqualsToken,'=='],[ts.SyntaxKind.ExclamationEqualsToken,'!='],[ts.SyntaxKind.BarBarToken,'||'],
  [ts.SyntaxKind.AmpersandAmpersandToken,'&&'],
]);
const assignOps = new Map([[ts.SyntaxKind.EqualsToken,'='],[ts.SyntaxKind.PlusEqualsToken,'+='],[ts.SyntaxKind.MinusEqualsToken,'-='],[ts.SyntaxKind.AsteriskEqualsToken,'*='],[ts.SyntaxKind.SlashEqualsToken,'/=']]);
const mathFns = new Map(Object.entries({min:'fmin',max:'fmax',abs:'fabs',sqrt:'sqrt',floor:'floor',ceil:'ceil',trunc:'trunc',exp:'exp',log:'log',log2:'log2',sin:'sin',cos:'cos',tan:'tan',pow:'pow',atan2:'atan2'}));
function cppNum(node){let n=Number(node.text);if(!Number.isFinite(n))throw new Error('nonfinite numeric literal');return Number.isInteger(n)?`${n}.0`:String(n);}
function failure(n,file,message){const at=file.getLineAndCharacterOfPosition(n.getStart(file));throw Error(`UNSUPPORTED ${file.fileName}:${at.line+1}:${at.character+1}: ${message}`);}
const sanitize = n => safe(n) ? n : (()=>{throw Error(`Unsafe identifier ${n}`)})();
const h = v => createHash('sha256').update(v).digest('hex');
const compileCache = new Map();
export function getStructuredCompilerCacheSize() { return compileCache.size; }

export function compileStructuredSource(source, options={}) {
  const digest=h(source);
  // General-purpose numeric lowering, never keyed to a function name or a domain.
  const fastIntegerRemainder = options.fastIntegerRemainder === true;
  const fastBoundedRemainder = options.fastBoundedRemainder === true;
  if (fastIntegerRemainder && fastBoundedRemainder) throw Error('Select at most one remainder optimization');
  const cacheKey = `${digest}:${fastIntegerRemainder ? 'integer-rem' : fastBoundedRemainder ? 'bounded-rem' : 'baseline'}`;
  if(compileCache.has(cacheKey))return compileCache.get(cacheKey);
  const file=ts.createSourceFile(options.filename??'generic-structured.ts',source,ts.ScriptTarget.ES2022,true,ts.ScriptKind.TS);
  if(file.parseDiagnostics.length)throw Error(ts.flattenDiagnosticMessageText(file.parseDiagnostics[0].messageText,' '));
  const interfaces=new Map(), declarations=[], globals=[], fns=new Map();
  for(const statement of file.statements){
    if(ts.isInterfaceDeclaration(statement)){
      const name=sanitize(statement.name.text), props=new Map();
      for(const m of statement.members){
        if(!ts.isPropertySignature(m)||!m.type||!m.name||!ts.isIdentifier(m.name)||m.questionToken)failure(m,file,'only required interface properties');
        props.set(sanitize(m.name.text),typeFrom(m.type));
      }
      const bases=(statement.heritageClauses??[]).flatMap(c=>c.types.map(t=>{
        if(!ts.isIdentifier(t.expression))failure(t,file,'interface extends must reference simple names');
        return sanitize(t.expression.text);
      }));
      interfaces.set(name,{name,props,bases});declarations.push({kind:'interface',name});
    }else if(ts.isFunctionDeclaration(statement)){
      if(!statement.name||!statement.body||statement.asteriskToken)failure(statement,file,'ordinary function body required');
      const name=sanitize(statement.name.text);
      if(fns.has(name))failure(statement,file,'duplicate function');
      fns.set(name,statement);declarations.push({kind:'function',name});
    }else if(ts.isVariableStatement(statement)){
      if(!(statement.declarationList.flags & ts.NodeFlags.Const)) failure(statement,file,'only top-level constants');
      globals.push(statement);declarations.push({kind:'global',statement});
    }else failure(statement,file,`top-level ${ts.SyntaxKind[statement.kind]}`);
  }
  if(fns.size===0)throw Error('No functions in supplied source');
  function typeFrom(node){
    if(node.kind===ts.SyntaxKind.NumberKeyword)return 'double';
    if(node.kind===ts.SyntaxKind.BooleanKeyword)return 'bool';
    if(ts.isTypeReferenceNode(node)){
      if(!ts.isIdentifier(node.typeName)||node.typeArguments?.length)failure(node,file,'unsupported type reference');
      return node.typeName.text==='Float64Array'?'a_span':sanitize(node.typeName.text);
    }
    failure(node,file,`type ${ts.SyntaxKind[node.kind]}`);
  }
  const structFields=new Map();
  function getFields(type,visiting=new Set()){
    if(structFields.has(type))return structFields.get(type);
    if(visiting.has(type))throw Error(`Cyclic interface ${type}`);
    const node=interfaces.get(type);if(!node)throw Error(`Unknown object type ${type}`);
    const all=new Map();visiting.add(type);
    for(const base of node.bases)for(const [k,v] of getFields(base,visiting))all.set(k,v);
    for(const [k,v] of node.props){if(all.has(k))throw Error(`Duplicate field ${k}`);all.set(k,v);}
    visiting.delete(type);structFields.set(type,all);return all;
  }
  for(const n of interfaces.keys())getFields(n);
  const globalNames=new Map(), fnSignatures=new Map();
  for(const decl of globals){
    for(const d of decl.declarationList.declarations){
      if(!ts.isIdentifier(d.name)||!d.initializer||globalNames.has(d.name.text))failure(d,file,'simple numeric top-level constant required');
      globalNames.set(sanitize(d.name.text),'double');
    }
  }
  for(const [name,fn] of fns){
    if(!fn.type)failure(fn,file,'explicit return type required');
    const args=fn.parameters.map(p=>{
      if(!ts.isIdentifier(p.name)||!p.type||p.initializer||p.questionToken||p.dotDotDotToken)failure(p,file,'required typed parameter');
      return {name:sanitize(p.name.text),type:typeFrom(p.type)};
    });
    fnSignatures.set(name,{name,args,result:typeFrom(fn.type)});
  }
  // Validate all references to named helper functions through this symbol table.
  let labelStack=[];
  // Source-driven closure conversion for CUDA extended lambdas.
  // CUDA forbids reference captures in __host__ __device__ lambdas.
  // Read-only outer values are captured by value; writes to outer
  // locals are redirected through explicitly captured pointers.
  const captureWritesStack=[];
  function rewriteCaptured(name) {
    const alias=captureWritesStack.at(-1)?.get(name);
    return alias ? `(*${alias})` : sanitize(name);
  }
  function externallyWrittenNames(body, outerNames, parameters) {
    // Collect the closure's own declarations before finding writes,
    // including declarations appearing after their first use.
    const own=new Set(parameters.map(p=>p.name));
    const visitDecl=node=>{
      if(ts.isFunctionDeclaration(node))return; // a different closure
      if(ts.isVariableDeclaration(node)){
        if(ts.isIdentifier(node.name))own.add(node.name.text);
        else if(ts.isObjectBindingPattern(node.name))for(const e of node.name.elements)
          if(ts.isIdentifier(e.name))own.add(e.name.text);
      }
      ts.forEachChild(node,visitDecl);
    };
    visitDecl(body);
    const modified=new Set();
    const visitWrites=node=>{
      if(ts.isFunctionDeclaration(node))return;
      let target;
      if(ts.isBinaryExpression(node)&&assignOps.has(node.operatorToken.kind)&&ts.isIdentifier(node.left))target=node.left.text;
      if((ts.isPrefixUnaryExpression(node)||ts.isPostfixUnaryExpression(node)) &&
         (node.operator===ts.SyntaxKind.PlusPlusToken||node.operator===ts.SyntaxKind.MinusMinusToken) &&
         ts.isIdentifier(node.operand))target=node.operand.text;
      if(target && outerNames.has(target) && !own.has(target))modified.add(target);
      ts.forEachChild(node,visitWrites);
    };
    visitWrites(body);
    return [...modified];
  }
  function compileExpr(e,names,depth=0){
    if(depth>100)failure(e,file,'expression too deep');
    const recur = x => compileExpr(x,names,depth+1);
    if(ts.isParenthesizedExpression(e))return `(${recur(e.expression)})`;
    if(ts.isNumericLiteral(e))return cppNum(e);
    if(e.kind===ts.SyntaxKind.TrueKeyword)return 'true';
    if(e.kind===ts.SyntaxKind.FalseKeyword)return 'false';
    if(ts.isIdentifier(e)){
      if(!names.has(e.text)&&!globalNames.has(e.text))failure(e,file,`unknown name ${e.text}`);
      return rewriteCaptured(e.text);
    }
    if(ts.isPrefixUnaryExpression(e)){
      const o=e.operator;
      if([ts.SyntaxKind.ExclamationToken,ts.SyntaxKind.MinusToken,ts.SyntaxKind.PlusToken].includes(o))return `(${ts.tokenToString(o)}${recur(e.operand)})`;
    }
    if(ts.isBinaryExpression(e)){
      if(e.operatorToken.kind===ts.SyntaxKind.QuestionQuestionToken){
        if(ts.isElementAccessExpression(e.left)&&e.left.argumentExpression){
          return `${recur(e.left.expression)}.getOr(${recur(e.left.argumentExpression)}, ${recur(e.right)})`;
        }
        failure(e,file,'nullish only supported for Float64Array[index]');
      }
      if(e.operatorToken.kind===ts.SyntaxKind.PercentToken)return `${fastBoundedRemainder ? "A_JS_BOUNDED_REMAINDER" : fastIntegerRemainder ? "A_JS_REMAINDER" : "fmod"}(${recur(e.left)}, ${recur(e.right)})`;
      if(e.operatorToken.kind===ts.SyntaxKind.AsteriskAsteriskToken)return `pow(${recur(e.left)}, ${recur(e.right)})`;
      const op=opMap.get(e.operatorToken.kind);
      if(op)return `(${recur(e.left)} ${op} ${recur(e.right)})`;
      failure(e,file,'assignment not valid within expression');
    }
    if(ts.isConditionalExpression(e))return `(${recur(e.condition)} ? ${recur(e.whenTrue)} : ${recur(e.whenFalse)})`;
    if(ts.isPropertyAccessExpression(e)){
      if(e.name.text==='length')return `(${recur(e.expression)}.length)`;
      return `${recur(e.expression)}.${sanitize(e.name.text)}`;
    }
    if(ts.isElementAccessExpression(e)&&e.argumentExpression)return `${recur(e.expression)}.getOr(${recur(e.argumentExpression)}, NAN)`;
    if(ts.isCallExpression(e)){
      if(ts.isIdentifier(e.expression)){
        const fname=e.expression.text;
        if(!fnSignatures.has(fname)&&!names.has(fname))failure(e,file,`unknown function ${fname}`);
        return `${sanitize(fname)}(${e.arguments.map(recur).join(', ')})`;
      }
      if(ts.isPropertyAccessExpression(e.expression)&&ts.isIdentifier(e.expression.expression)&&e.expression.expression.text==='Math'){
        const m=e.expression.name.text;
        if(m==='round'){
          if(e.arguments.length!==1)failure(e,file,'Math.round arity');
          return `floor(${recur(e.arguments[0])} + 0.5)`;
        }
        if(!mathFns.has(m))failure(e,file,`Math.${m} unsupported`);
        const arity=['pow','atan2'].includes(m)?2:1;
        if(m==='min'||m==='max'){
          if(e.arguments.length<2)failure(e,file,'Math.min/max need 2+ arguments');
          return e.arguments.slice(1).reduce((acc,a)=>`${mathFns.get(m)}(${acc},${recur(a)})`,recur(e.arguments[0]));
        }
        if(e.arguments.length!==arity)failure(e,file,'Math arity mismatch');
        return `${mathFns.get(m)}(${e.arguments.map(recur).join(', ')})`;
      }
      failure(e,file,'unsupported call target');
    }
    if(ts.isObjectLiteralExpression(e))failure(e,file,'object literal only permitted as a typed return');
    failure(e,file,`expression ${ts.SyntaxKind[e.kind]}`);
  }
  function compileReturnObject(e,names,type){
    if(!ts.isObjectLiteralExpression(e))failure(e,file,`return ${type} requires object literal`);
    const shape=getFields(type), fields=new Map();
    for(const p of e.properties){
      if(!ts.isPropertyAssignment(p)&&!ts.isShorthandPropertyAssignment(p))failure(p,file,'simple object property');
      if(!ts.isIdentifier(p.name))failure(p,file,'simple object property name');
      const n=p.name.text;
      if(fields.has(n)||!shape.has(n))failure(p,file,'duplicate/unknown return field');
      fields.set(n,ts.isPropertyAssignment(p)?compileExpr(p.initializer,names):compileExpr(p.name,names));
    }
    if(fields.size!==shape.size)failure(e,file,`return ${type} fields incomplete`);
    return `${type}{${[...shape.keys()].map(n=>fields.get(n)).join(', ')}}`;
  }
  function compileStatements(node,names,returnType,depth=0){
    if(depth>100)failure(node,file,'statement nesting too deep');
    if(ts.isBlock(node)){
      const inner=new Map(names);
      return `{\n${node.statements.map(s=>compileStatements(s,inner,returnType,depth+1)).join('\n')}\n}`;
    }
    if(ts.isVariableStatement(node)){
      let out=[];
      for(const d of node.declarationList.declarations){
        if(ts.isObjectBindingPattern(d.name)){
          if(!d.initializer)failure(d,file,'object destructuring requires initializer');
          const rhs=compileExpr(d.initializer,names);
          let type='';if(ts.isIdentifier(d.initializer))type=names.get(d.initializer.text);
          if(!structFields.has(type))failure(d,file,'object destructuring requires typed interface');
          const f=getFields(type);
          for(const prop of d.name.elements){
            if(prop.dotDotDotToken||prop.initializer||!ts.isIdentifier(prop.name)||prop.propertyName&&!ts.isIdentifier(prop.propertyName))failure(prop,file,'simple destructuring only');
            const name=sanitize(prop.name.text), key=prop.propertyName?.text??name;
            if(names.has(name)||!f.has(key))failure(prop,file,'duplicate/unknown destructured variable');
            names.set(name,f.get(key));out.push(`const auto ${name} = ${rhs}.${sanitize(key)};`);
          }
        }else{
          if(!ts.isIdentifier(d.name))failure(d,file,'unsupported variable binding');
          const name=sanitize(d.name.text);
          if(names.has(name))failure(d,file,'shadowing disabled');
          const type=d.type?typeFrom(d.type):inferType(d.initializer,names);
          if(!d.initializer && !d.type)failure(d,file,'declaration needs annotation or initializer');
          if(type==='a_span' && !d.initializer)failure(d,file,'span requires initializer');
          let value=d.initializer?` = ${compileExpr(d.initializer,names)}`:'';
          names.set(name,type);
          out.push(`${node.declarationList.flags&ts.NodeFlags.Const?'const ':''}${type} ${name}${value};`);
        }
      }
      return out.join('\n');
    }
    if(ts.isReturnStatement(node)){
      if(!node.expression)failure(node,file,'explicit return value required');
      const value=structFields.has(returnType)?compileReturnObject(node.expression,names,returnType):compileExpr(node.expression,names);
      return `return ${value};`;
    }
    if(ts.isIfStatement(node))return `if (${compileExpr(node.expression,names)}) ${compileStatements(node.thenStatement,new Map(names),returnType,depth+1)}${node.elseStatement?` else ${compileStatements(node.elseStatement,new Map(names),returnType,depth+1)}`:''}`;
    if(ts.isExpressionStatement(node)&&ts.isBinaryExpression(node.expression)){
      const e=node.expression,op=assignOps.get(e.operatorToken.kind);
      if(!op||!ts.isIdentifier(e.left)||!names.has(e.left.text))failure(node,file,'assignment to declared local only');
      return `${rewriteCaptured(e.left.text)} ${op} ${compileExpr(e.right,names)};`;
    }
    if(ts.isForStatement(node)){
      if(!node.initializer||!ts.isVariableDeclarationList(node.initializer)||node.initializer.declarations.length!==1)failure(node,file,'one for-index declaration required');
      const d=node.initializer.declarations[0];if(!ts.isIdentifier(d.name)||!d.initializer)failure(d,file,'for-index initializer required');
      const index=sanitize(d.name.text),inner=new Map(names);
      if(inner.has(index))failure(d,file,'for index shadows local');
      inner.set(index,'double');
      if(!node.condition||!node.incrementor)failure(node,file,'bounded for condition/increment required');
      let increment='';
      const inc=node.incrementor;
      if(ts.isPostfixUnaryExpression(inc)||ts.isPrefixUnaryExpression(inc)){
        if(!ts.isIdentifier(inc.operand)||inc.operand.text!==index||![ts.SyntaxKind.PlusPlusToken,ts.SyntaxKind.MinusMinusToken].includes(inc.operator))failure(inc,file,'invalid for step');
        increment=`${index}${inc.operator===ts.SyntaxKind.PlusPlusToken?'++':'--'}`;
      }else failure(inc,file,'for increment must be ++/--');
      // Variable bounds are accepted. Generated watchdog prevents infinite loops.
      const start=compileExpr(d.initializer,names),condition=compileExpr(node.condition,inner);
      const body=compileStatements(node.statement,inner,returnType,depth+1);
      return `for(double ${index} = ${start}, a_loop_guard_${depth}=0.0; (${condition}); ${increment}, ++a_loop_guard_${depth}) { if (a_loop_guard_${depth} >= A_LOOP_LIMIT) A_TRAP(); ${body} }`;
    }
    if(ts.isFunctionDeclaration(node)){
      if(!node.name||!node.body||!node.type)failure(node,file,'nested named function needs body/type');
      const name=sanitize(node.name.text);if(names.has(name))failure(node,file,'duplicate nested function');
      const args=node.parameters.map(p=>{
        if(!ts.isIdentifier(p.name)||!p.type||p.initializer)failure(p,file,'nested function requires typed arguments');
        return {name:sanitize(p.name.text),type:typeFrom(p.type)};
      });
      const returnTypeLocal=typeFrom(node.type);
      const locals=new Map(names);
      for(const p of args){if(locals.has(p.name))failure(node,file,'shadowed lambda parameter');locals.set(p.name,p.type);}
      names.set(name,`function:${returnTypeLocal}`);
      const written=externallyWrittenNames(node.body,names,args);
      const aliases=new Map(written.map(n=>[n,`a_capture_write_${name}_${n}`]));
      captureWritesStack.push(aliases);
      let fnBody;
      try { fnBody=compileStatements(node.body,locals,returnTypeLocal,depth+1); }
      finally { captureWritesStack.pop(); }
      const pointerDecls=written.map(n=>`auto* ${aliases.get(n)} = &${rewriteCaptured(n)};`).join('\n');
      return `${pointerDecls}\nauto ${name} = [=] A_HD (${args.map(p=>`${p.type} ${p.name}`).join(', ')}) -> ${returnTypeLocal} ${fnBody};`;
    }
    if(ts.isLabeledStatement(node)){
      if(labelStack.includes(node.label.text))failure(node,file,'duplicate label');
      labelStack.push(node.label.text);const inner=compileStatements(node.statement,new Map(names),returnType,depth+1);labelStack.pop();
      return `do ${inner} while(false);`;
    }
    if(ts.isBreakStatement(node)){
      if(!node.label||labelStack.at(-1)!==node.label.text)failure(node,file,'break label must match innermost labelled block');
      return 'break;';
    }
    if(ts.isContinueStatement(node)){
      if(node.label)failure(node,file,'labelled continue unsupported');
      return 'continue;';
    }
    if(ts.isThrowStatement(node)&&ts.isNewExpression(node.expression)){
      if(!ts.isIdentifier(node.expression.expression)||node.expression.expression.text!=='Error')failure(node,file,'only Error validation throws');
      return 'A_TRAP();';
    }
    failure(node,file,`statement ${ts.SyntaxKind[node.kind]}`);
  }
  function inferType(init,names){
    if(!init)return 'double';
    if(init.kind===ts.SyntaxKind.TrueKeyword||init.kind===ts.SyntaxKind.FalseKeyword)return 'bool';
    if(ts.isIdentifier(init))return names.get(init.text)??globalNames.get(init.text)??'double';
    if(ts.isCallExpression(init)&&ts.isIdentifier(init.expression))return fnSignatures.get(init.expression.text)?.result??'double';
    if(ts.isPropertyAccessExpression(init)&&ts.isIdentifier(init.expression)){
      const type=names.get(init.expression.text);return structFields.get(type)?.get(init.name.text)??'double';
    }
    return 'double';
  }
  const structs=[...interfaces.keys()].map(name=>{
    const shape=interfaces.get(name);
    if(shape.bases.length>1)throw Error(`Multiple interface inheritance not yet supported: ${name}`);
    const base=shape.bases.length?` : ${shape.bases[0]}`:'';
    const members=[...shape.props].map(([field,type])=>`  ${type} ${field};`).join('\n');
    return `struct ${name}${base} {\n${members}\n};`;
  });
  const globalCode=globals.flatMap(d=>d.declarationList.declarations.map(decl=>{
    const names=new Map(globalNames);
    return `constexpr double ${decl.name.text} = ${compileExpr(decl.initializer,names)};`;
  }));
  const funcs=[...fns].map(([name,fn])=>{
    const signature=fnSignatures.get(name), names=new Map(globalNames);
    for(const p of signature.args){if(names.has(p.name))failure(fn,file,'argument shadows global');names.set(p.name,p.type);}
    const body=compileStatements(fn.body,names,signature.result);
    return `A_HD static inline ${signature.result} ${name}(${signature.args.map(p=>`${p.type} ${p.name}`).join(', ')}) ${body}`;
  });
  const prelude=`#include <cmath>\n#include <cstdint>\n#include <cstdlib>\n#include <limits>\n#ifdef __CUDACC__\n#define A_HD __host__ __device__\n#else\n#define A_HD\n#endif\n#define A_LOOP_LIMIT 1000000.0\nA_HD static inline void A_TRAP(){\n#ifdef __CUDA_ARCH__\nasm("trap;");\n#else\nstd::abort();\n#endif\n}\nstruct a_span {\n const double* data; int length;\n A_HD double getOr(double position, double fallback) const {\n if(position<0 || position>=length || position!=floor(position)) return fallback;\n return data[(int)position];\n }\n};`;
  // For integer-valued double operands in the signed 32-bit range, C++
  // signed integer remainder is exactly the JS remainder for this subset.
  // Preserve signed zero and use fmod for ALL other numeric cases.
  const integerRemainderHelper = `A_HD static inline double A_JS_REMAINDER(double a, double b) {
   if (a >= -2147483647.0 && a <= 2147483647.0 &&
       b >= -2147483647.0 && b <= 2147483647.0 &&
       b != 0.0 && floor(a) == a && floor(b) == b) {
     int32_t ai = (int32_t)a, bi = (int32_t)b;
     int32_t r = ai % bi;
     return r == 0 ? copysign(0.0, a) : (double)r;
   }
   return fmod(a, b);
 }`;
  // General IEEE-754 remainder strength reduction: for positive divisor b
  // and 0 <= a < b, the remainder is exactly a; for b <= a < 2*b,
  // Sterbenz's lemma guarantees a-b is exact in binary floating point.
  // For all other inputs (including negative zero, NaN, and negative
  // divisors), retain the standard JS-compatible fmod operation.
  const boundedRemainderHelper = `A_HD static inline double A_JS_BOUNDED_REMAINDER(double a, double b) {
    if (b > 0.0 && a >= 0.0) {
      if (a < b) return a;
      if (a < (2.0 * b)) return a - b;
    }
    return fmod(a, b);
  }`;
  const optimizedPrelude = fastBoundedRemainder
    ? prelude.replace('struct a_span {', `${boundedRemainderHelper}\nstruct a_span {`)
    : fastIntegerRemainder
      ? prelude.replace('struct a_span {', `${integerRemainderHelper}\nstruct a_span {`)
      : prelude;
  const cpp=[optimizedPrelude,...structs,...globalCode,...funcs].join('\n\n');
  // The returned source is a fully expanded translation of AST nodes.
  const compiled=Object.freeze({hash:digest,cpp,functions:[...fnSignatures.values()],interfaces:Object.fromEntries([...interfaces].map(([n,v])=>[n,{bases:v.bases,ownFields:Object.fromEntries(v.props)}])),structs:Object.fromEntries([...structFields].map(([n,m])=>[n,Object.fromEntries(m)])),globalCount:globalNames.size});
  compileCache.set(cacheKey,compiled);
  return compiled;
}
