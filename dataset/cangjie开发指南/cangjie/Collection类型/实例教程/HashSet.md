class HashSet<T> where T <: Hashable & Equatable<T>
public class HashSet<T> <: Set<T> where T <: Hashable & Equatable<T> {
    public init()
    public init(elements: Collection<T>)
    public init(elements: Array<T>)
    public init(capacity: Int64)
    public init(size: Int64, initElement: (Int64) -> T)
}
功能：基于 HashMap 实现的 Set 接口的实例。

HashSet中的元素是无序的，不允许有重复元素。当我们向HashSet中添加元素时，HashSet会根据元素的哈希值来确定该元素在哈希表中的位置。

提示：

HashSet 是基于 HashMap 实现的，因此 HashSet 的容量、内存布局、时间性能等都和 HashMap 相同。

父类型：

Set<T>
prop size
public prop size: Int64
功能：返回此 HashSet 的元素个数。

类型：Int64

init(Int64, Int64 -> T)
public init(size: Int64, initElement: (Int64) -> T)
功能：通过传入的函数元素个数 size 和函数规则来构造 HashSet。构造出的 HashSet 的容量受 size 大小影响。

参数：

size: Int64 - 初始化函数中元素的个数。
initElement: (Int64) ->T - 初始化函数规则。
异常：

IllegalArgumentException - 如果 size 小于 0，抛出异常。
init()
public init()
功能：构造一个空的 HashSet ，初始容量为 16 。

init(Array<T>)
public init(elements: Array<T>)
功能：使用传入的数组构造 HashSet。该构造函数根据传入数组 elements 的 size 设置 HashSet 的容量。

参数：

elements: Array<T> - 初始化 HashSet 的数组。
init(Collection<T>)
public init(elements: Collection<T>)
功能：使用传入的集合构造 HashSet。该构造函数根据传入集合 elements 的 size 设置 HashSet 的容量。

参数：

elements: Collection<T> - 初始化 HashSet 的集合。
init(Int64)
public init(capacity: Int64)
功能：使用传入的容量构造一个 HashSet。

参数：

capacity: Int64 - 初始化容量大小。
异常：

IllegalArgumentException - 如果 capacity 小于 0，抛出异常。
func capacity()
public func capacity(): Int64
功能：返回此 HashSet 的内部数组容量大小。

注意:

容量大小不一定等于 HashSet 的 size。

返回值：

Int64 - 返回此 HashSet 的内部数组容量大小。
func clear()
public func clear(): Unit
功能：从此 HashSet 中移除所有元素。

func clone()
public func clone(): HashSet<T>
功能：克隆 HashSet。

返回值：

HashSet<T> - 返回克隆到的 HashSet。
func contains(T)
public func contains(element: T): Bool
功能：判断 HashSet 是否包含指定元素。

参数：

element: T - 指定的元素。
返回值：

Bool - 如果包含指定元素，则返回 true；否则，返回 false。
func containsAll(Collection<T>)
public func containsAll(elements: Collection<T>): Bool
功能：判断 HashSet 是否包含指定 Collection 中的所有元素。

参数：

elements: Collection<T> - 指定的元素集合。
返回值：

Bool - 如果此 HashSet 包含 Collection 中的所有元素，则返回 true；否则，返回 false。
func isEmpty()
public func isEmpty(): Bool
功能：判断 HashSet 是否为空。

返回值：

Bool - 如果为空，则返回 true；否则，返回 false。
func iterator()
public func iterator(): Iterator<T>
功能：返回此 HashSet 的迭代器。

返回值：

Iterator<T> - 返回此 HashSet 的迭代器。
示例：

使用示例见 HashSet 的 put/iterator/remove 函数。

func put(T)
public func put(element: T): Bool
功能：将指定的元素添加到 HashSet 中, 若添加的元素在 HashSet 中存在, 则添加失败。

参数：

element: T - 指定的元素。
返回值：

Bool - 如果添加成功，则返回 true；否则，返回 false。
示例：

使用示例见 HashSet 的 put/iterator/remove 函数。

func putAll(Collection<T>)
public func putAll(elements: Collection<T>): Unit
功能：添加 Collection 中的所有元素至此 HashSet 中，如果元素存在，则不添加。

参数：

elements: Collection<T> - 需要被添加的元素的集合。
func remove(T)
public func remove(element: T): Bool
功能：如果指定元素存在于此 HashSet 中，则将其移除。

参数：

element: T - 需要被移除的元素。
返回值：

Bool - true，表示移除成功；false，表示移除失败。
示例：

使用示例见 HashSet 的 put/iterator/remove 函数。

func removeAll(Collection<T>)
public func removeAll(elements: Collection<T>): Unit
功能：移除此 HashSet 中那些也包含在指定 Collection 中的所有元素。

参数：

elements: Collection<T> - 需要从此 HashSet 中移除的元素的集合。
func removeIf((T) -> Bool)
public func removeIf(predicate: (T) -> Bool): Unit
功能：传入 lambda 表达式，如果满足 true 条件，则删除对应的元素。

参数：

predicate: (T) ->Bool - 是否删除元素的判断条件。
异常：

ConcurrentModificationException - 当 predicate 中增删或者修改 HashSet 内键值对时，抛出异常。
func reserve(Int64)
public func reserve(additional: Int64): Unit
功能：将 HashSet 扩容 additional 大小，当 additional 小于等于零时，不发生扩容；当 HashSet 剩余容量大于等于 additional 时，不发生扩容；当 HashSet 剩余容量小于 additional 时，取（原始容量的1.5倍向下取整）与（additional + 已使用容量）中的最大值进行扩容。

参数：

additional: Int64 - 将要扩容的大小。
异常：

OverflowException - 当additional + 已使用容量超过Int64.Max时，抛出异常。
func retainAll(Set<T>)
public func retainAll(elements: Set<T>): Unit
功能：从此 HashSet 中保留 Set 中的元素。

参数：

elements: Set<T> - 需要保留的 Set。
func subsetOf(Set<T>)
public func subsetOf(other: Set<T>): Bool
功能：检查该集合是否为其他 Set 的子集。

参数：

other: Set<T> - 传入集合，此函数将判断当前集合是否为 other 的子集。
返回值：

Bool - 如果该 Set 是指定 Set 的子集，则返回 true；否则返回 false。
func toArray()
public func toArray(): Array<T>
功能：返回一个包含容器内所有元素的数组。

返回值：

Array<T> - T 类型数组。
extend<T> HashSet<T> <: Equatable<HashSet<T>>
extend<T> HashSet<T> <: Equatable<HashSet<T>>
功能：为 HashSet<T> 类型扩展 Equatable<HashSet<T>> 接口，支持判等操作。

父类型：

Equatable<HashSet<T>>
operator func ==(HashSet<T>)
public operator func ==(that: HashSet<T>): Bool
功能：判断当前实例与参数指向的 HashSet<T> 实例是否相等。

两个 HashSet<T> 相等指的是其中包含的元素完全相等。

参数：

that: HashSet<T> - 被比较的对象。
返回值：

Bool - 如果相等，则返回 true，否则返回 false。
operator func !=(HashSet<T>)
public operator func !=(that: HashSet<T>): Bool
功能：判断当前实例与参数指向的 HashSet<T> 实例是否不等。

参数：

that: HashSet<T> - 被比较的对象。
返回值：

Bool - 如果不等，则返回 true，否则返回 false。
extend<T> HashSet<T> <: ToString where T <: ToString
extend<T> HashSet<T> <: ToString where T <: ToString
功能：为 HashSet<T> 扩展 ToString 接口，支持转字符串操作。

父类型：

ToString
func toString()
public func toString(): String
功能：将当前 HashSet<T> 实例转换为字符串。

该字符串包含 HashSet<T> 内每个元素的字符串表示，形如："[elem1, elem2, elem3]"。

返回值：

String - 转换得到的字符串。