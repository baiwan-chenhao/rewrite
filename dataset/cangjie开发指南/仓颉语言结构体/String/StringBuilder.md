class StringBuilder
public class StringBuilder <: ToString {
    public init()
    public init(str: String)
    public init(r: Rune, n: Int64)
    public init(value: Array<Rune>)
    public init(capacity: Int64)
}
功能：该类主要用于字符串的构建。

StringBuilder 在字符串的构建上效率高于 String：

在功能上支持传入多个类型的值，该类将自动将其转换为 String 类型对象，并追加到构造的字符串中。
在性能上使用动态扩容算法，减少内存申请频率，构造字符串的速度更快，占用内存资源通常更少。
注意：

StringBuilder 仅支持 UTF-8 编码的字符数据。

父类型：

ToString
prop capacity
public prop capacity: Int64
功能：获取 StringBuilder 实例此时能容纳字符串的长度，该值会随扩容的发生而变大。

类型：Int64

prop size
public prop size: Int64
功能：获取 StringBuilder 实例中字符串长度。

类型：Int64

init()
public init()
功能：构造一个初始容量为 32 的空 StringBuilder 实例。

init(Array<Rune>)
public init(value: Array<Rune>)
功能：使用参数 value 指定的字符数组初始化一个 StringBuilder 实例，该实例的初始容量为 value 大小，初始内容为 value 包含的字符内容。

参数：

value: Array<Rune> - 初始化 StringBuilder 实例的字符数组。
init(Int64)
public init(capacity: Int64)
功能：使用参数 capacity 指定的容量初始化一个空 StringBuilder 实例，该实例的初始容量为 value 大小，初始内容为若干 \0 字符。

参数：

capacity: Int64 - 初始化 StringBuilder 的字节容量，取值范围为 (0, Int64.Max]。
异常：

IllegalArgumentException - 当参数 capacity 的值小于等于 0 时，抛出异常。
init(Rune, Int64)
public init(r: Rune, n: Int64)
功能：使用 n 个 r 字符初始化 StringBuilder 实例，该实例的初始容量为 n，初始内容为 n 个 r 字符。

参数：

r: Rune - 初始化 StringBuilder 实例的字符。
n: Int64 - 字符 r 的数量，取值范围为 [0, Int64.Max]。
异常：

IllegalArgumentException - 当参数 n 小于 0 时，抛出异常。
init(String)
public init(str: String)
功能：根据指定初始字符串构造 StringBuilder 实例，该实例的初始容量为指定字符串的大小，初始内容为指定字符串。

参数：

str: String - 初始化 StringBuilder 实例的字符串。
func append(Array<Rune>)
public func append(runeArr: Array<Rune>): Unit
功能：在 StringBuilder 末尾插入一个 Rune 数组中所有字符。

参数：

runeArr: Array<Rune> - 插入的 Rune 数组。
func append<T>(Array<T>) where T <: ToString
public func append<T>(val: Array<T>): Unit where T <: ToString
功能：在 StringBuilder 末尾插入参数 val 指定的 Array<T> 的字符串表示，类型 T 需要实现 ToString 接口。

参数：

val: Array<T> - 插入的 Array<T> 类型实例。
func append(Bool)
public func append(b: Bool): Unit
功能：在 StringBuilder 末尾插入参数 b 的字符串表示。

参数：

b: Bool - 插入的 Bool 类型的值。
func append(CString)
public func append(cstr: CString): Unit
功能：在 StringBuilder 末尾插入参数 cstr 指定 CString 中的内容。

参数：

cstr: CString - 插入的 CString。
func append(Float16)
public func append(n: Float16): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: Float16 - 插入的 Float16 类型的值。
func append(Float32)
public func append(n: Float32): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: Float32 - 插入的 Float32 类型的值。
func append(Float64)
public func append(n: Float64): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: Float64 - 插入的 Float64 类型的值。
func append(Int16)
public func append(n: Int16): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: Int16 - 插入的 Int16 类型的值。
func append(Int32)
public func append(n: Int32): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: Int32 - 插入的 Int32 类型的值。
func append(Int64)
public func append(n: Int64): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: Int64 - 插入的 Int64 类型的值。
func append(Int8)
public func append(n: Int8): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: Int8 - 插入的 Int8 类型的值。
func append(Rune)
public func append(r: Rune): Unit
功能：在 StringBuilder 末尾插入参数 r 指定的字符。

参数：

r: Rune - 插入的字符。
func append(String)
public func append(str: String): Unit
功能：在 StringBuilder 末尾插入参数 str 指定的字符串。

参数：

str: String - 插入的字符串。
func append(StringBuilder)
public func append(sb: StringBuilder): Unit
功能：在 StringBuilder 末尾插入参数 sb 指定的 StringBuilder 中的内容。

参数：

sb: StringBuilder - 插入的 StringBuilder 实例。
func append<T>(T) where T <: ToString
public func append<T>(v: T): Unit where T <: ToString
功能：在 StringBuilder 末尾插入参数 v 指定 T 类型的字符串表示，类型 T 需要实现 ToString 接口。

参数：

v: T - 插入的 T 类型实例。
func append(UInt16)
public func append(n: UInt16): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: UInt16 - 插入的 UInt16 类型的值。
func append(UInt32)
public func append(n: UInt32): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: UInt32 - 插入的 UInt32 类型的值。
func append(UInt64)
public func append(n: UInt64): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: UInt64 - 插入的 UInt64 类型的值。
func append(UInt8)
public func append(n: UInt8): Unit
功能：在 StringBuilder 末尾插入参数 n 的字符串表示。

参数：

n: UInt8 - 插入的 UInt8 类型的值。
func appendFromUtf8(Array<Byte>)
public func appendFromUtf8(arr: Array<Byte>): Unit
功能：在 StringBuilder 末尾插入参数 arr 指向的字节数组。

该函数要求参数 arr 符合 UTF-8 编码，如果不符合，将抛出异常。

参数：

arr: Array<Byte> - 插入的字节数组。
异常：

IllegalArgumentException - 当字节数组不符合 utf8 编码规则时，抛出异常。
func appendFromUtf8Unchecked(Array<Byte>)
public unsafe func appendFromUtf8Unchecked(arr: Array<Byte>): Unit
功能：在 StringBuilder 末尾插入参数 arr 指向的字节数组。

相较于 appendFromUtf8 函数，它并没有针对于字节数组进行 UTF-8 相关规则的检查，所以它所构建的字符串并不一定保证是合法的，甚至出现非预期的异常，如果不是某些场景下的速度考虑，请优先使用安全的 appendFromUtf8 函数。

参数：

arr: Array<Byte> - 插入的字节数组。
func reserve(Int64)
public func reserve(additional: Int64): Unit
功能：将 StringBuilder 扩容 additional 大小。

当 additional 小于等于零，或剩余容量大于等于 additional 时，不发生扩容；当剩余容量小于 additional 时，扩容至当前容量的 1.5 倍（向下取整）与 size + additional 的最大值。

参数：

additional: Int64 - 指定 StringBuilder 的扩容大小。
func reset(?Int64)
public func reset(capacity!: Option<Int64> = None): Unit
功能：清空当前 StringBuilder，并将容量重置为 capacity 指定的值。

参数：

capacity!: Option<Int64> - 重置后 StringBuilder 实例的容量大小，取值范围为 None 和 (Some(0), Some(Int64.Max)]，默认值 None 表示采用默认大小容量（32）。
异常：

IllegalArgumentException - 当参数 capacity 的值小于等于 0 时，抛出异常。
func toString()
public func toString(): String
功能：获取 StringBuilder 实例中的字符串。

注意：

该函数不会将字符串数据进行拷贝。

返回值：

String - StringBuilder 实例中的字符串。