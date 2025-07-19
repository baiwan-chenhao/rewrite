func toAsciiLower()
public func toAsciiLower(): String
功能：将该字符串中所有 Ascii 大写字母转化为 Ascii 小写字母。

返回值：

String - 转换后的新字符串。
func toAsciiTitle()
public func toAsciiTitle(): String
功能：将该字符串标题化。

该函数只转换 Ascii 英文字符，当该英文字符是字符串中第一个字符或者该字符的前一个字符不是英文字符，则该字符大写，其他英文字符小写。

返回值：

String - 转换后的新字符串。
func toAsciiUpper()
public func toAsciiUpper(): String
功能：将该字符串中所有 Ascii 小写字母转化为 Ascii 大写字母。

返回值：

String - 转换后的新字符串。
func toRuneArray()
public func toRuneArray(): Array<Rune>
功能：获取字符串的 Rune 数组。如果原字符串为空字符串，则返回空数组。

返回值：

Array<Rune> - 字符串的 Rune 数组。
func toString()
public func toString(): String
功能：获得字符串本身。

返回值：

String - 返回字符串本身。
func trimAscii()
public func trimAscii(): String
功能：去除原字符串开头结尾以 whitespace 字符组成的子字符串。

whitespace 的 unicode 码点范围为 [0009, 000D] 和 [0020]。

返回值：

String - 转换后的新字符串。
func trimAsciiLeft()
public func trimAsciiLeft(): String
功能：去除原字符串开头以 whitespace 字符组成的子字符串。

返回值：

String - 转换后的新字符串。
func trimAsciiRight()
public func trimAsciiRight(): String
功能：去除原字符串结尾以 whitespace 字符组成的子字符串。

返回值：

String - 转换后的新字符串。
func trimLeft(String)
public func trimLeft(prefix: String): String
功能：去除字符串的 prefix 前缀。

参数：

prefix: String - 待去除的前缀。
返回值：

String - 转换后的新字符串。
func trimRight(String)
public func trimRight(suffix: String): String
功能：去除字符串的 suffix 后缀。

参数：

suffix: String - 待去除的后缀。
返回值：

String - 转换后的新字符串。
func tryGet(Int64)
public func tryGet(index: Int64): Option<Byte>
功能：返回字符串下标 index 对应的 UTF-8 编码字节值。

参数：

index: Int64 - 要获取的字节值的下标。
返回值：

Option<Byte> - 获取得到下标对应的 UTF-8 编码字节值，当 index 小于 0 或者大于等于字符串长度，则返回 Option<Byte>.None。
operator func !=(String)
public const operator func !=(right: String): Bool
功能：判断两个字符串是否不相等。

参数：

right: String - 待比较的 String 实例。
返回值：

Bool - 不相等返回 true，相等返回 false。
operator func *(Int64)
public const operator func *(count: Int64): String
功能：原字符串重复 count 次。

参数：

count: Int64 - 原字符串重复的次数。
返回值：

String - 返回重复 count 次后的新字符串。
operator func +(String)
public const operator func +(right: String): String
功能：两个字符串相加，将 right 字符串拼接在原字符串的末尾。

参数：

right: String - 待追加的字符串。
返回值：

String - 返回拼接后的字符串。
operator func <(String)
public const operator func <(right: String): Bool
功能：判断两个字符串大小。

参数：

right: String - 待比较的字符串。
返回值：

Bool - 原字符串字典序小于 right 时，返回 true，否则返回 false。
operator func <=(String)
public const operator func <=(right: String): Bool
功能：判断两个字符串大小。

参数：

right: String - 待比较的字符串。
返回值：

Bool - 原字符串字典序小于或等于 right 时，返回 true，否则返回 false。
operator func ==(String)
public const operator func ==(right: String): Bool
功能：判断两个字符串是否相等。

参数：

right: String - 待比较的字符串。
返回值：

Bool - 相等返回 true，不相等返回 false。
operator func >(String)
public const operator func >(right: String): Bool
功能：判断两个字符串大小。

参数：

right: String - 待比较的字符串。
返回值：

Bool - 原字符串字典序大于 right 时，返回 true，否则返回 false。
operator func >=(String)
public const operator func >=(right: String): Bool
功能：判断两个字符串大小。

参数：

right: String - 待比较的字符串。
返回值：

Bool - 原字符串字典序大于或等于 right 时，返回 true，否则返回 false。
operator func [](Int64)
public const operator func [](index: Int64): Byte
功能：返回指定索引 index 处的 UTF-8 编码字节。

参数：

index: Int64 - 要获取 UTF-8 编码字节的下标。
返回值：

Byte - 获取得到下标对应的 UTF-8 编码字节。
异常：

IndexOutOfBoundsException - 如果 index 小于 0 或大于等于字符串长度，抛出异常。
operator func [](Range<Int64>)
public const operator func [](range: Range<Int64>): String
功能：根据给定区间获取当前字符串的切片。

注意：

如果参数 range 是使用 Range 构造函数构造的 Range 实例，有如下行为：
start 的值就是构造函数传入的值本身，不受构造时传入的 hasStart 的值的影响。
hasEnd 为 false 时，end 值不生效，且不受构造时传入的 isClosed 的值的影响，该字符串切片取到原字符串最后一个元素。
range 的步长只能为 1。
参数：

range: Range<Int64> - 切片的区间。
返回值：

String - 字符串切片。
异常：

IndexOutOfBoundsException - 如果切片范围超过原字符串边界，抛出异常。
IllegalArgumentException - 如果 range.step 不等于 1或者范围起止点不是字符边界，抛出异常。