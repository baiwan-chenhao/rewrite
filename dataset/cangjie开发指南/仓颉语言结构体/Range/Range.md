struct Range<T> where T <: Countable<T> & Comparable<T> & Equatable<T>
public struct Range<T> <: Iterable<T> where T <: Countable<T> & Comparable<T> & Equatable<T> {
    public let end: T
    public let hasEnd: Bool
    public let hasStart: Bool
    public let isClosed: Bool
    public let start: T
    public let step: Int64
    public const init(start: T, end: T, step: Int64, hasStart: Bool, hasEnd: Bool, isClosed: Bool)
}
功能：该类是区间类型，用于表示一个拥有固定范围和步长的 T 的序列，要求 T 是可数的，有序的。

区间类型有对应的字面量表示，其格式为：

左闭右开区间：start..end : step，它表示一个从 start 开始，以 step 为步长，到 end（不包含 end）为止的区间。
左闭右闭区间：start..=end : step，它表示一个从 start 开始，以 step 为步长，到 end（包含 end）为止的区间。
注意：

当 step > 0 且 start >= end，或者 step < 0 且 start <= end 时，该 Range 实例将是一个空区间。
当 step > 0 且 start > end，或者 step < 0 且 start < end 时，该 Range 实例将是一个空区间。
父类型：

Iterable<T>
let end
public let end: T
功能：表示结束值。

类型：T

let hasEnd
public let hasEnd: Bool
功能：表示是否包含结束值。

类型：Bool

let hasStart
public let hasStart: Bool
功能：表示是否包含开始值。

类型：Bool

let isClosed
public let isClosed: Bool
功能：表示区间开闭情况，为 true 表示左闭右闭，为 false 表示左闭右开。

类型：Bool

let start
public let start: T
功能：表示开始值。

类型：T

let step
public let step: Int64
功能：表示步长。

类型：Int64

init(T, T, Int64, Bool, Bool, Bool)
public const init(start: T, end: T, step: Int64, hasStart: Bool, hasEnd: Bool, isClosed: Bool)
功能：使用该构造函数创建 Range 序列。

参数：

start: T - 开始值。
end: T - 结束值。
step: Int64 - 步长，取值不能为 0。
hasStart: Bool - 是否有开始值。
hasEnd: Bool - 是否有结束值。
isClosed: Bool - true 代表左闭右闭，false 代表左闭右开。
异常：

IllegalArgumentException - 当 step 等于 0 时, 抛出异常。
func isEmpty()
public const func isEmpty(): Bool
功能：判断该区间是否为空。

返回值：

Bool - 如果为空，返回 true，否则返回 false。
func iterator()
public func iterator(): Iterator<T>
功能：获取当前区间的迭代器。

返回值：

Iterator<T> - 当前区间的迭代器。
extend<T> Range<T> <: Equatable<Range<T>> where T <: Countable<T> & Comparable<T> & Equatable<T>
extend<T> Range<T> <: Equatable<Range<T>> where T <: Countable<T> & Comparable<T> & Equatable<T>
功能：为 Range<T> 类型扩展 Equatable<Range<T>> 接口。

父类型：

Equatable<Range<T>>
operator func !=(Range<T>)
public operator func !=(that: Range<T>): Bool
功能：判断两个 Range 是否不相等。

参数：

that: Range<T> - 待比较的 Range 实例。
返回值：

Bool - true 代表不相等，false 代表相等。
operator func ==(Range<T>)
public operator func ==(that: Range<T>): Bool
功能：判断两个 Range 实例是否相等。

两个 Range 实例相等指的是它们表示同一个区间，即 start、end、step、isClosed 值相等。

参数：

that: Range<T> - 待比较的 Range 实例。
返回值：

Bool - true 代表相等，false 代表不相等。
extend<T> Range<T> <: Hashable where T <: Hashable & Countable<T> & Comparable<T> & Equatable<T>
extend<T> Range<T> <: Hashable where T <: Hashable & Countable<T> & Comparable<T> & Equatable<T>
功能：为 Range 类型扩展 Hashable 接口，支持计算哈希值。

父类型：

Hashable
func hashCode()
public func hashCode(): Int64
功能：获取哈希值，该值为 start、end、step、isClosed 的组合哈希运算结果。

返回值：

Int64 - 哈希值。