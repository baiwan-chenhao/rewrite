class LinkedList<T>
public class LinkedList<T> <: Collection<T> {
    public init()
    public init(elements: Collection<T>)
    public init(elements: Array<T>)
    public init(size: Int64, initElement: (Int64)-> T)
}
功能：实现双向链表的数据结构。

双向链表是一种常见的数据结构，它由一系列节点组成，每个节点都包含两个指针，一个指向前一个节点，另一个指向后一个节点。这种结构允许在任何一个节点上进行双向遍历，即可以从头节点开始向后遍历，也可以从尾节点开始向前遍历。

LinkedList 不支持并发操作，并且对集合中元素的修改不会使迭代器失效，只有在添加和删除元素的时候会使迭代器失效。

父类型：

Collection<T>
prop first
public prop first: ?T
功能：链表中第一个元素的值，如果是空链表则返回 None。

类型：?T

prop last
public prop last: ?T
功能：链表中最后一个元素的值，如果是空链表则返回 None。

类型：?T

prop size
public prop size: Int64
功能：链表中的元素数量。

类型：Int64

init
public init()
功能：构造一个空的链表。

init(Array<T>)
public init(elements: Array<T>)
功能：按照数组的遍历顺序构造一个包含指定集合元素的 LinkedList 实例。

参数：

elements: Array<T> - 将要放入此链表中的元素数组。
init(Collection<T>)
public init(elements: Collection<T>)
功能：按照集合迭代器返回元素的顺序构造一个包含指定集合元素的链表。

参数：

elements: Collection<T> - 将要放入此链表中的元素集合。
init(Int64, (Int64)-> T)
public init(size: Int64, initElement: (Int64)-> T)
功能：创建一个包含 size 个元素，且第 n 个元素满足 (Int64)-> T 条件的链表。

参数：

size: Int64 - 要创建的链表元素数量。
initElement: (Int64) ->T - 元素的初始化参数。
异常：

IllegalArgumentException - 如果指定的链表长度小于 0 则抛此异常。
func append(T)
public func append(element: T): LinkedListNode<T>
功能：在链表的尾部位置添加一个元素，并且返回该元素的节点。

参数：

element: T - 要添加到链表中的元素。
返回值：

LinkedListNode<T> - 指向该元素的节点。
func clear()
public func clear(): Unit
功能：删除链表中的所有元素。

func firstNode()
public func firstNode(): Option<LinkedListNode<T>>
功能：获取链表中的第一个元素的节点。

返回值：

Option < LinkedListNode<T>> - 第一个元素的节点，如果链表为空链表则返回 None。
func insertAfter(LinkedListNode<T>,T)
public func insertAfter(node: LinkedListNode<T>, element: T): LinkedListNode<T>
功能：在链表中指定节点的后面插入一个元素，并且返回该元素的节点。

参数：

node: LinkedListNode<T> - 指定的节点。
element: T - 要添加到链表中的元素。
返回值：

LinkedListNode<T> - 指向被插入元素的节点。
异常：

IllegalArgumentException - 如果指定的节点不属于该链表，则抛此异常。
func insertBefore(LinkedListNode<T>,T)
public func insertBefore(node: LinkedListNode<T>, element: T): LinkedListNode<T>
功能：在链表中指定节点的前面插入一个元素，并且返回该元素的节点。

参数：

node: LinkedListNode<T> - 指定的节点。
element: T - 要添加到链表中的元素。
返回值：

LinkedListNode<T> - 指向被插入元素的节点。
异常：

IllegalArgumentException - 如果指定的节点不属于该链表，则抛此异常。
func isEmpty()
public func isEmpty(): Bool
功能：返回此链表是否为空链表的判断。

返回值：

Bool - 如果此链表中不包含任何元素，返回 true。
func iterator()
public func iterator(): Iterator<T>
功能：返回当前集合中元素的迭代器，其顺序是从链表的第一个节点到链表的最后一个节点。

返回值：

Iterator<T> - 当前集合中元素的迭代器。
func lastNode()
public func lastNode(): Option<LinkedListNode<T>>
功能：获取链表中的最后一个元素的节点。

返回值：

Option < LinkedListNode<T>> - 最后一个元素的节点，如果链表为空链表则返回 None。
func nodeAt(Int64)
public func nodeAt(index: Int64): Option<LinkedListNode<T>>
功能：获取链表中的第 index 个元素的节点，编号从 0 开始。

该函数的时间复杂度为 O(n)。

参数：

index: Int64 - 指定获取第 index 个元素的节点。
返回值：

Option < LinkedListNode<T>> - 编号为 index 的节点，如果没有则返回 None。
func popFirst()
public func popFirst() : ?T
功能：移除链表的第一个元素，并返回该元素的值。

返回值：

?T - 被删除的元素的值，若链表为空则返回 None。
func popLast()
public func popLast() : ?T
功能：移除链表的最后一个元素，并返回该元素的值。

返回值：

?T - 被删除的元素的值，若链表为空则返回 None。
func prepend(T)
public func prepend(element: T): LinkedListNode<T>
功能：在链表的头部位置插入一个元素，并且返回该元素的节点。

参数：

element: T - 要添加到链表中的元素。
返回值：

LinkedListNode<T> - 指向该元素的节点。
func remove(LinkedListNode<T>)
public func remove(node: LinkedListNode<T>): T
功能：删除链表中指定节点。

参数：

node: LinkedListNode<T> - 要被删除的节点。
返回值：

T - 被删除的节点的值。
异常：

IllegalArgumentException - 如果指定的节点不属于该链表，则抛此异常。
func removeIf((T)-> Bool)
public func removeIf(predicate: (T)-> Bool): Unit
功能：删除此链表中满足给定 lambda 表达式或函数的所有元素。

参数：

predicate: (T) ->Bool - 对于要删除的元素，返回值为 true。
func reverse()
public func reverse(): Unit
功能：反转此链表中的元素顺序。

func splitOff(LinkedListNode<T>)
public func splitOff(node: LinkedListNode<T>): LinkedList<T>
功能：从指定的节点 node 开始，将链表分割为两个链表，如果分割成功，node 不在当前的链表内，而是作为首个节点存在于新的链表内部。

参数：

node: LinkedListNode<T> - 要分割的位置。
返回值：

LinkedList<T> - 原链表分割后新产生的链表。
异常：

IllegalArgumentException - 如果指定的节点不属于该链表，则抛此异常。
func toArray()
public func toArray(): Array<T>
功能：返回一个数组，数组包含该链表中的所有元素，并且顺序与链表的顺序相同。

返回值：

Array<T> - T 类型数组。
extend<T> LinkedList<T> <: Equatable<LinkedList<T>> where T <: Equatable<T>
extend<T> LinkedList<T> <: Equatable<LinkedList<T>> where T <: Equatable<T>
功能：为 LinkedList<T> 类型扩展 Equatable<LinkedList<T>> 接口，支持判等操作。

父类型：

Equatable<LinkedList<T>>
operator func ==(LinkedList<T>)
public operator func ==(right: LinkedList<T>): Bool
功能：判断当前实例与参数指向的 LinkedList<T> 实例是否相等。

两个 LinkedList<T> 相等指的是其中包含的元素完全相等。

参数：

right: HashSet<T> - 被比较的对象。
返回值：

Bool - 如果相等，则返回 true，否则返回 false。
operator func !=(LinkedList<T>)
public operator func !=(right: LinkedList<T>): Bool
功能：判断当前实例与参数指向的 LinkedList<T> 实例是否不等。

参数：

right: LinkedList<T> - 被比较的对象。
返回值：

Bool - 如果不等，则返回 true，否则返回 false。
extend<T> LinkedList<T> <: ToString where T <: ToString
extend<T> LinkedList<T> <: ToString where T <: ToString
功能：为 LinkedList<T> 扩展 ToString 接口，支持转字符串操作。

父类型：

ToString
func toString()
public func toString(): String
功能：将当前 LinkedList<T> 实例转换为字符串。

该字符串包含 LinkedList<T> 内每个元素的字符串表示，形如："[elem1, elem2, elem3]"。

返回值：

String - 转换得到的字符串。