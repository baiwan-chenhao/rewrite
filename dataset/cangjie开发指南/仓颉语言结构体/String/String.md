struct String
public struct String <: Collection<Byte> & Equatable<String> & Comparable<String> & Hashable & ToString {
    public static const empty: String
    public const init()
    public init(value: Array<Rune>)
    public init(value: Collection<Rune>)
}
功能：该结构体表示仓颉字符串，提供了构造、查找、拼接等一系列字符串操作。

注意：

String 类型仅支持 UTF-8 编码。

父类型：

Collection<Byte>
Equatable<String>
Comparable<String>
Hashable
ToString
static const empty
public static const empty: String = String()
功能：创建一个空的字符串并返回。

类型：String

prop size
public prop size: Int64
功能：获取字符串 UTF-8 编码后的字节长度。

类型：Int64

init()
public const init()
功能：构造一个空的字符串。

init(Array<Rune>)
public init(value: Array<Rune>)
功能：根据字符数组构造一个字符串，字符串内容为数组中的所有字符。

参数：

value: Array<Rune> - 根据该字符数组构造字符串。
init(Collection<Rune>)
public init(value: Collection<Rune>)
功能：据字符集合构造一个字符串，字符串内容为集合中的所有字符。

参数：

value: Collection<Rune> - 根据该字符集合构造字符串。
static func fromUtf8(Array<UInt8>)
public static func fromUtf8(utf8Data: Array<UInt8>): String
功能：根据 UTF-8 编码的字节数组构造一个字符串。

参数：

utf8Data: Array<UInt8> - 根据该字节数组构造字符串。
返回值：

String - 构造的字符串。
异常：

IllegalArgumentException - 入参不符合 utf-8 序列规则，抛出异常。
static func fromUtf8Unchecked(Array<UInt8>)
public static unsafe func fromUtf8Unchecked(utf8Data: Array<UInt8>): String
功能：根据字节数组构造一个字符串。

相较于 fromUtf8 函数，它并没有针对于字节数组进行 UTF-8 相关规则的检查，所以它所构建的字符串并不一定保证是合法的，甚至出现非预期的异常，如果不是某些场景下的性能考虑，请优先使用安全的 fromUtf8 函数。

参数：

utf8Data: Array<UInt8> - 根据该字节数组构造字符串。
返回值：

String - 构造的字符串。
static func join(Array<String>, String)
public static func join(strArray: Array<String>, delimiter!: String = String.empty): String
功能：连接字符串列表中的所有字符串，以指定分隔符分隔。

参数：

strArray: Array<String> - 需要被连接的字符串数组，当数组为空时，返回空字符串。
delimiter!: String - 用于连接的中间字符串，其默认值为 String.empty。
返回值：

String - 连接后的新字符串。
func clone()
public func clone(): String
功能：返回原字符串的拷贝。

返回值：

String - 拷贝得到的新字符串。
func compare(String)
public func compare(str: String): Ordering
功能：按字典序比较当前字符串和参数指定的字符串。

参数：

str: String - 被比较的字符串。
返回值：

Ordering - 返回 enum 值 Ordering 表示结果，Ordering.GT 表示当前字符串字典序大于 str 字符串，Ordering.LT 表示当前字符串字典序小于 str 字符串，Ordering.EQ 表示两个字符串字典序相等。
异常：

IllegalArgumentException - 如果两个字符串的原始数据中存在无效的 UTF-8 编码，抛出异常。
func contains(String)
public func contains(str: String): Bool
功能：判断原字符串中是否包含字符串 str。

参数：

str: String - 待搜索的字符串。
返回值：

Bool - 如果字符串 str 在原字符串中，返回 true，否则返回 false。特别地，如果 str 字符串长度为 0，返回 true。
func count(String)
public func count(str: String): Int64
功能：返回子字符串 str 在原字符串中出现的次数。

参数：

str: String - 被搜索的子字符串。
返回值：

Int64 - 出现的次数，当 str 为空字符串时，返回原字符串中 Rune 的数量加一。
func endsWith(String)
public func endsWith(suffix: String): Bool
功能：判断原字符串是否以 suffix 字符串为后缀结尾。

参数：

suffix: String - 被判断的后缀字符串。
返回值：

Bool - 如果字符串 str 是原字符串的后缀，返回 true，否则返回 false，特别地，如果 str 字符串长度为 0，返回 true。
func getRaw()
public unsafe func getRaw(): CPointerHandle<UInt8>
功能：获取当前 String 的原始指针，用于和C语言交互，使用完后需要 releaseRaw 函数释放该指针。

注意：

getRaw 与 releaseRaw 之间仅可包含简单的 foreign C 函数调用等逻辑，不构造例如 CString 等的仓颉对象，否则可能造成不可预知的错误。

返回值：

CPointerHandle<UInt8> - 当前字符串的原始指针实例。
func hashCode()
public func hashCode(): Int64
功能：获取字符串的哈希值。

返回值：

Int64 - 返回字符串的哈希值。
func indexOf(Byte)
public func indexOf(b: Byte): Option<Int64>
功能：获取指定字节 b 第一次出现的在原字符串内的索引。

参数：

b: Byte - 待搜索的字节。
返回值：

Option<Int64> - 如果原字符串中包含指定字节，返回其第一次出现的索引，如果原字符串中没有此字节，返回 None。
func indexOf(Byte, Int64)
public func indexOf(b: Byte, fromIndex: Int64): Option<Int64>
功能：从原字符串指定索引开始搜索，获取指定字节第一次出现的在原字符串内的索引。

参数：

b: Byte - 待搜索的字节。
fromIndex: Int64 - 以指定的索引 fromIndex 开始搜索。
返回值：

Option<Int64> - 如果搜索成功，返回指定字节第一次出现的索引，否则返回 None。特别地，当 fromIndex 小于零，效果同 0，当 fromIndex 大于等于原字符串长度，返回 None。
func indexOf(String)
public func indexOf(str: String): Option<Int64>
功能：返回指定字符串 str 在原字符串中第一次出现的起始索引。

参数：

str: String - 待搜索的字符串。
返回值：

Option<Int64> - 如果原字符串包含 str 字符串，返回其第一次出现的索引，如果原字符串中没有 str 字符串，返回 None。
func indexOf(String, Int64)
public func indexOf(str: String, fromIndex: Int64): Option<Int64>
功能：从原字符串 fromIndex 索引开始搜索，获取指定字符串 str 第一次出现的在原字符串的起始索引。

参数：

str: String - 待搜索的字符串。
fromIndex: Int64 - 以指定的索引 fromIndex 开始搜索。
返回值：

Option<Int64> - 如果搜索成功，返回 str 第一次出现的索引，否则返回 None。特别地，当 str 是空字符串时，如果fromIndex 大于 0，返回 None，否则返回 Some(0)。当 fromIndex 小于零，效果同 0，当 fromIndex 大于等于原字符串长度返回 None。
func isAscii()
public func isAscii(): Bool
功能：判断字符串是否是一个 Ascii 字符串，如果字符串为空或没有 Ascii 以外的字符，则返回 true。

返回值：

Bool - 是则返回 true，不是则返回 false。
func isAsciiBlank()
public func isAsciiBlank(): Bool
功能：判断字符串是否为空或者字符串中的所有 Rune 都是 ascii 码的空白字符（包括：0x09、0x10、0x11、0x12、0x13、0x20）。

返回值：

Bool - 如果是返回 true，否则返回 false。
func isEmpty()
public func isEmpty(): Bool
功能：判断原字符串是否为空字符串。

返回值：

Bool - 如果为空返回 true，否则返回 false。
func iterator()
public func iterator(): Iterator<Byte>
功能：获取字符串的 UTF-8 编码字节迭代器，可用于支持 for-in 循环。

返回值：

Iterator<Byte> - 字符串的 UTF-8 编码字节迭代器。