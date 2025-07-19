class ArrayList<T>

public class ArrayList<T> <: Collection<T> {
    public init()
    public init(capacity: Int64)
    public init(size: Int64, initElement: (Int64) -> T)
    public init(elements: Array<T>)
    public init(elements: Collection<T>)
}
功能：提供可变长度的数组的功能。

ArrayList 是一种线性的动态数组，与 Array 不同，它可以根据需要自动调整大小，并且在创建时不需要指定大小。

说明：

当向动态数组中添加元素时，如果数组已满，则会重新分配更大的内存空间，并将原有的元素复制到新的内存空间中。

动态数组的优点是可以节省内存空间，并且可以根据需要自动调整大小，因此非常适合需要频繁添加或删除元素的情况。但是，动态数组的缺点是在重新分配内存空间时可能会导致性能下降，因此在使用动态数组时需要考虑这一点。

父类型：

Collection<T>
prop size
public prop size: Int64
功能：返回此 ArrayList 中的元素个数。

类型：Int64

init()
public init()
功能：构造一个初始容量大小为默认值16的ArrayList。

init(Array<T>)
public init(elements: Array<T>)
功能：构造一个包含指定数组中所有元素的 ArrayList。

注意：

当 T 的类型是 Int64 时，此构造函数的变长参数语法糖版本可能会和 public init(Int64) 产生歧义，比如 ArrayList<Int64>(8, 9) 是构造一个包含两个元素的 ArrayList， 而 ArrayList<Int64>(8) 是构造一个容量为 8 的 ArrayList。

参数：

elements: Array<T> - 传入数组。
init(Collection<T>)
public init(elements: Collection<T>)
功能：构造一个包含指定集合中所有元素的 ArrayList。这些元素按照集合的迭代器返回的顺序排列。

参数：

elements: Collection<T> - 传入集合。
init(Int64)
public init(capacity: Int64)
功能：构造一个初始容量为指定大小的 ArrayList。

参数：

capacity: Int64 - 指定的初始容量大小。
异常：

IllegalArgumentException - 如果参数的大小小于 0 则抛出异常。
init(Int64, (Int64) -> T)
public init(size: Int64, initElement: (Int64) -> T)
功能：构造具有指定初始元素个数和指定规则函数的 ArrayList。该构造函数根据参数 size 设置 ArrayList 的容量。

参数：

size: Int64 - 初始化函数元素个数。
initElement: (Int64) ->T - 传入初始化函数。
异常：

IllegalArgumentException - 如果 size 小于 0 则抛出异常。
func append(T)
public func append(element: T): Unit
功能：将指定的元素附加到此 ArrayList 的末尾。

参数：

element: T - 插入的元素，类型为 T。
示例：

使用示例见 ArrayList 的 append/insert 函数。

func appendAll(Collection<T>)
public func appendAll(elements: Collection<T>): Unit
功能：将指定集合中的所有元素附加到此 ArrayList 的末尾。

函数会按照迭代器顺序遍历入参中的集合，并且将所有元素插入到此 ArrayList 的尾部。

参数：

elements: Collection<T> - 需要插入的元素的集合。
func capacity()
public func capacity(): Int64
功能：返回此 ArrayList 的容量大小。

返回值：

Int64 - 此 ArrayList 的容量大小。
func clear()
public func clear(): Unit
功能：从此 ArrayList 中删除所有元素。

示例：

使用示例见 ArrayList 的 remove/clear/slice 函数。

func clone()
public func clone(): ArrayList<T>
功能：返回此ArrayList实例的拷贝(浅拷贝)。

返回值：

ArrayList<T> - 返回新 ArrayList<T>。
func get(Int64)
public func get(index: Int64): ?T
功能：返回此 ArrayList 中指定位置的元素。

参数：

index: Int64 - 要返回的元素的索引。
返回值：

?T - 返回指定位置的元素，如果 index 大小小于 0 或者大于等于 ArrayList 中的元素数量，返回 None。
示例：

使用示例见 ArrayList 的 get/set 函数。

func getRawArray()
public unsafe func getRawArray(): Array<T>
功能：返回 ArrayList 的原始数据。

注意：

这是一个 unsafe 的接口，使用处需要在 unsafe 上下文中。

原始数据是指 ArrayList 底层实现的数组，其大小大于等于 ArrayList 中的元素数量，且索引大于等于 ArrayList 大小的位置中可能包含有未初始化的元素，对其进行访问可能会产生未定义的行为。

返回值：

Array<T> - ArrayList 的底层原始数据。
func insert(Int64, T)
public func insert(index: Int64, element: T): Unit
功能：在此 ArrayList 中的指定位置插入指定元素。

参数：

index: Int64 - 插入元素的目标索引。
element: T - 要插入的 T 类型元素。
异常：

IndexOutOfBoundsException - 当 index 超出范围时，抛出异常。
示例：

使用示例见 ArrayList 的 append/insert 函数。

func insertAll(Int64, Collection<T>)
public func insertAll(index: Int64, elements: Collection<T>): Unit
功能：从指定位置开始，将指定集合中的所有元素插入此 ArrayList。

函数会按照迭代器顺序遍历入参中的集合，并且将所有元素插入到指定位置。

参数：

index: Int64 - 插入集合的目标索引。
elements: Collection<T> - 要插入的 T 类型元素集合。
异常：

IndexOutOfBoundsException - 当 index 超出范围时，抛出异常。
示例：

使用示例见 ArrayList 的 remove/clear/slice 函数。

func isEmpty()
public func isEmpty(): Bool
功能：判断 ArrayList 是否为空。

返回值：

Bool - 如果为空，则返回 true，否则，返回 false。
func iterator()
public func iterator(): Iterator<T>
功能：返回此 ArrayList 中元素的迭代器。

返回值：

Iterator<T> - ArrayList 中元素的迭代器。
func prepend(T)
public func prepend(element: T): Unit
功能：在起始位置，将指定元素插入此 ArrayList。

参数：

element: T - 插入 T 类型元素。
func prependAll(Collection<T>)
public func prependAll(elements: Collection<T>): Unit
功能：从起始位置开始，将指定集合中的所有元素插入此 ArrayList。

函数会按照迭代器顺序遍历入参中的集合，并且将所有元素插入到指定位置。

参数：

elements: Collection<T> - 要插入的 T 类型元素集合。
func remove(Int64)
public func remove(index: Int64): T
功能：删除此 ArrayList 中指定位置的元素。

参数：

index: Int64 - 被删除元素的索引。
返回值：

T - 被移除的元素。
异常：

IndexOutOfBoundsException - 当 index 超出范围时，抛出异常。
示例：

使用示例见 ArrayList 的 remove/clear/slice 函数。