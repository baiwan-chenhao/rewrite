const 函数和常量求值
常量求值允许某些特定形式的表达式在编译时求值，可以减少程序运行时需要的计算。本章主要介绍常量求值的使用方法与规则。

const 变量
const 变量是一种特殊的变量，它以关键字 const 修饰，定义在编译时完成求值，并且在运行时不可改变的变量。例如，下面的例子定义了万有引力常数 G：

const G = 6.674e-11
const 变量可以省略类型标注，但是不可省略初始化表达式。const 变量可以是全局变量，局部变量，静态成员变量。但是 const 变量不能在扩展中定义。const 变量可以访问对应类型的所有实例成员，也可以调用对应类型的所有非 mut 实例成员函数。

下例定义了一个 struct，记录行星的质量和半径，同时定义了一个 const 成员函数 gravity 用来计算该行星对距离为 r 质量为 m 的物体的万有引力：

struct Planet {
    const Planet(let mass: Float64, let radius: Float64) {}

    const func gravity(m: Float64, r: Float64) {
        G * mass * m / r**2
    }
}

main() {
    const myMass = 71.0
    const earth = Planet(5.972e24, 6.378e6)
    println(earth.gravity(myMass, earth.radius))
}
编译执行得到地球对地面上一个质量为 71 kg 的成年人的万有引力：

695.657257
const 变量初始化后该类型实例的所有成员都是 const 的（深度 const，包含成员的成员），因此不能被用于左值。

main() {
    const myMass = 71.0
    myMass = 70.0 // Error, cannot assign to immutable value
}
const 上下文与 const 表达式
const 上下文是指 const 变量初始化表达式，这些表达式始终在编译时求值。因此需要对 const 上下文中允许的表达式加以限制，避免修改全局状态、I/O 等副作用，确保其可以在编译时求值。

const 表达式具备了可以在编译时求值的能力。满足如下规则的表达式是 const 表达式：

数值类型、Bool、Unit、Rune、String 类型的字面量（不包含插值字符串）。
所有元素都是 const 表达式的 Array 字面量（不能是 Array 类型，可以使用 VArray 类型），tuple 字面量。
const 变量，const 函数形参，const 函数中的局部变量。
const 函数，包含使用 const 声明的函数名、符合 const 函数要求的 lambda、以及这些函数返回的函数表达式。
const 函数调用（包含 const 构造函数），该函数的表达式必须是 const 表达式，所有实参必须都是 const 表达式。
所有参数都是 const 表达式的 enum 构造器调用，和无参数的 enum 构造器。
数值类型、Bool、Unit、Rune、String 类型的算术表达式、关系表达式、位运算表达式，所有操作数都必须是 const 表达式。
if、match、try、控制转移表达式（包含 return、break、continue、throw）、is、as。这些表达式内的表达式必须都是 const 表达式。
const 表达式的成员访问（不包含属性的访问），tuple 的索引访问。
const init 和 const 函数中的 this 和 super 表达式。
const 表达式的 const 实例成员函数调用，且所有实参必须都是 const 表达式。
const 函数
const 函数是一类特殊的函数，这些函数具备了可以在编译时求值的能力。在 const 上下文中调用这种函数时，这些函数会在编译时执行计算。而在其它非 const 上下文，const 函数会和普通函数一样在运行时执行。

下例是一个计算平面上两点距离的 const 函数，distance 中使用 let 定义了两个局部变量 dx 和 dy：

struct Point {
    const Point(let x: Float64, let y: Float64) {}
}

const func distance(a: Point, b: Point) {
    let dx = a.x - b.x
    let dy = a.y - b.y
    (dx**2 + dy**2)**0.5
}

main() {
    const a = Point(3.0, 0.0)
    const b = Point(0.0, 4.0)
    const d = distance(a, b)
    println(d)
}
编译运行输出：

5.000000
需要注意：

const 函数声明必须使用 const 修饰。
全局 const 函数和 static const 函数中只能访问 const 声明的外部变量，包含 const 全局变量、const 静态成员变量，其它外部变量都不可访问。const init 函数和 const 实例成员函数除了能访问 const 声明的外部变量，还可以访问当前类型的实例成员变量。
const 函数中的表达式都必须是 const 表达式，const init 函数除外。
const 函数中可以使用 let、const 声明新的局部变量。但不支持 var。
const 函数中的参数类型和返回类型没有特殊规定。如果该函数调用的实参不符合 const 表达式要求，那这个函数调用不能作为 const 表达式使用，但仍然可以作为普通表达式使用。
const 函数不一定都会在编译时执行，例如可以在非 const 函数中运行时调用。
const 函数与非 const 函数重载规则一致。
数值类型、Bool、Unit、Rune、String 类型 和 enum 支持定义 const 实例成员函数。
对于 struct 和 class，只有定义了 const init 才能定义 const 实例成员函数。class 中的 const 实例成员函数不能是 open 的。struct 中的 const 实例成员函数不能是 mut 的。
另外，接口中也可以定义 const 函数，但会受到以下规则限制：

接口中的 const 函数，实现类型必须也用 const 函数才算实现接口。
接口中的非 const 函数，实现类型使用 const 或非 const 函数都算实现接口。
接口中的 const 函数与接口的 static 函数一样，只有在该接口作为泛型约束的时候，受约束的泛型变元或变量才能使用这些 const 函数。
在下面的例子中，在接口 I 里定义了两个 const 函数，类 A 实现了接口 I，泛型函数 g 的形参类型上界是 I。

interface I {
    const func f(): Int64
    const static func f2(): Int64
}

class A <: I {
    public const func f() { 0 }
    public const static func f2() { 1 }
    const init() {}
}

const func g<T>(i: T) where T <: I {
    return i.f() + T.f2()
}

main() {
    println(g(A()))
}
编译执行上述代码，输出结果为：

1
const init
如果一个 struct 或 class 定义了 const 构造器，那么这个 struct/class 实例可以用在 const 表达式中。

如果当前类型是 class，则不能具有 var 声明的实例成员变量，否则不允许定义 const init 。如果当前类型具有父类，当前的 const init 必须调用父类的 const init（可以显式调用或者隐式调用无参const init），如果父类没有 const init 则报错。
当前类型的实例成员变量如果有初始值，初始值必须要是 const 表达式，否则不允许定义 const init。
const init 内可以使用赋值表达式对实例成员变量赋值，除此以外不能有其它赋值表达式。
const init 与 const 函数的区别是 const init 内允许对实例成员变量进行赋值（需要使用赋值表达式）。