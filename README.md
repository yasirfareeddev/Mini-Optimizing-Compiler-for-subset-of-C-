# Mini Optimizing Compiler for subset of C++

A single-file educational compiler implementing the complete compilation pipeline for a C++ subset language with emphasis on optimization techniques including constant folding and dead code elimination.

## Overview

This project implements a modular compiler that transforms source code through all major compilation phases:

- **Lexical Analysis**: Tokenization using regex-based lexer with comprehensive error reporting
- **Syntax Analysis**: Recursive-descent LL(1) parser constructing Abstract Syntax Trees (AST)
- **Semantic Analysis**: Type checking, symbol table management, and type inference for `auto` declarations
- **Intermediate Code Generation**: Three Address Code (TAC) representation with temporary variables
- **Optimization**: Constant folding and dead code elimination passes
- **Code Generation**: Assembly-like instruction output
- **Execution**: Built-in interpreter for validating generated code

The compiler demonstrates how compile-time analysis can significantly reduce instruction count while preserving program semantics.

## Features

### Language Support
- Data types: `int`, `float`, `char`, `string`, `auto` (with type inference)
- Arithmetic operators: `+`, `-`, `*`, `/`, `%`
- Relational operators: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Control flow: `if`/`else` statements, `while` loops
- Output operations: `cout << expression` with chained outputs
- Variable declarations with optional initialization
- Function definitions (parameterless only, e.g., `int main()`)

### Optimizations
- **Constant Folding**: Evaluates constant expressions at compile time
  ```c
  int x = 5 + 3 * 2;  // Folded to: x = 11


* **Dead Code Elimination**: Removes assignments to temporaries that are never used after folding

  ```c
  // Original TAC:        // After DCE:
  // t1 = 7 * 3           // (eliminated)
  // t2 = 4 * 2           // (eliminated)
  // temp1 = t1 + t2      // temp1 = 29
  ```

## Project Structure

```
mini-optimizing-compiler/
├── backend/
│   ├── app.py          # Flask REST API server
│   └── compiler.py     # Complete compiler implementation (lexer → parser → semantic → TAC → optimization)
├── docs/
│   ├── Design_Implementation_Optimization.pdf      # Academic research paper (peer-reviewed format)
│   └── Compiler_Construction_Project_Documentation.pdf  # Course project documentation with implementation details
├── index.html          # Web interface (frontend entry point)
├── script.js           # Frontend logic and API integration
├── style.css           # Styling for web interface
├── LICENSE             # MIT License
├── README.md           # Project documentation
└── .gitignore          # Excluded files configuration
```

## Setup and Execution

### Prerequisites

* Python 3.10 or higher
* pip package manager
* VS Code with Live Server extension (for frontend)

### Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   https://github.com/yasirfareeddev/Mini-Optimizing-Compiler-for-subset-of-C-.git
   cd Mini-Optimizing-Compiler-for-subset-of-C-
   ```

2. Install backend dependencies:

   ```bash
   cd backend
   pip install flask flask-cors
   ```

3. Start the backend server:

   ```bash
   python3 app.py
   ```

   The server will run at `http://localhost:5000`

4. Launch the frontend:

   * Open the project root in VS Code
   * Right-click `index.html` and select "Open with Live Server"
   * The interface will open at `http://127.0.0.1:5500`

5. Usage:

   * Enter source code in the left editor panel
   * Click "Compile" to process the code
   * View results in the right panel: tokens, AST, unoptimized TAC, optimized TAC, assembly, output, and errors

### Command-Line Usage (Alternative)

For direct execution without the web interface:

```bash
cd backend
python3
>>> from compiler import compile_source
>>> result = compile_source('int x = 5 + 3 * 2; cout << x;')
>>> print("\\n".join(str(t) for t in result['optimized_tac']))
```

## Example Program

```cpp
int main() {
    int a = 10;
    int b = 20;
    int c = (a + b) * 3;
    int result = c / 2;
    cout << result;
}
```

**Optimization Impact**:

* Unoptimized TAC: 7 instructions with 3 temporaries (`t1`, `t2`, `t3`)
* Optimized TAC: 4 instructions with constant folding applied (`c = 90`, `result = 45`)
* Dead code elimination removes unused temporaries after folding

## Limitations

### Language Constraints

* Function definitions support only parameterless signatures (e.g., `int main()`)
* No support for arrays, pointers, structs, or user-defined types
* Limited string operations (only concatenation with `+` operator)
* No support for `break` or `continue` statements despite token recognition
* Character literals support basic escape sequences only

### Optimization Constraints

* Constant folding applies only to expressions with literal operands
* Dead code elimination targets only unused temporary variables (`t1`, `t2`, etc.)
* No unreachable code elimination for control flow blocks
* No loop optimizations or strength reduction
* No constant propagation across variable assignments
* No algebraic simplifications (e.g., `x * 1` → `x`)

### Semantic Constraints

* Type inference for `auto` works only with immediate initializers
* No implicit conversion from `float` to `int` (only `int` to `float` permitted)
* No support for boolean literals beyond integer comparisons (`0` = false, non-zero = true)
* Limited error recovery; single error may prevent full compilation

### Technical Constraints

* Parser requires explicit semicolons; no automatic insertion
* No support for preprocessor directives (`#include`, `#define`)
* Input operations (`cin`) are tokenized but not implemented in execution
* Assembly output is conceptual; not executable on real hardware

## Academic Context

This project was developed as a course assignment for Compiler Construction (BSCS 6th Semester) at Iqra Univeristy ( C&T Department ). The implementation emphasizes educational clarity over production performance, with each compiler phase implemented in isolated, readable modules suitable for academic demonstration.

## Research Paper
Read our academic paper on this project: "Design, Implementation and Optimization of Mini Compiler for subset of C++" in document folder.

## Contributors

* Yasir Fareed (22245)
* Umer Qazi (22068)
* Huma Ibrar (22292)

Submitted to: Sir Safdar
Course: Compiler Construction
Semester: Spring 2025

## License

This project is for educational purposes only. Redistribution or commercial use requires explicit permission from the authors.


