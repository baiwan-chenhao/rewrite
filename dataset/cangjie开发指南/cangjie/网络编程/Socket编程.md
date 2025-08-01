Socket 编程
仓颉的 Socket 编程指的是基于传输层协议实现网络传输数据包的功能。

在可靠传输场景下，仓颉分别启动客户端套接字和服务端套接字。客户端套接字必须指定将要连接的远端地址，可选择性地绑定本端地址，在连接成功后，才可以收发报文。而服务端套接字必须绑定本端地址，在绑定成功后，才可以收发报文。

在不可靠传输场景下，套接字无需区分客户端和服务端，仓颉分别启动两个套接字进行数据传输。套接字必须绑定本端地址，绑定成功后，才可以收发报文。并且，套接字也可选择性地指定远端连接地址，指定后将仅接受指定的远端地址的报文，同时在 send 时无需指定远端地址，报文将发送至成功连接的地址。

Tcp 编程
Tcp 作为一种常见的可靠传输协议，以 Tcp 类型套接字举例，仓颉在可靠传输场景下的可参考的编程模型如下：

创建服务端套接字，并指定本端绑定地址。
执行绑定。
执行 accept 动作，将阻塞等待，直到获取到一个客户端套接字连接。
同步创建客户端套接字，并指定远端的待连接的地址。
执行连接。
连接成功后，服务端会在 accept 接口返回一个新的套接字，此时服务端可以通过此套接字进行读写操作，即收发报文。客户端则可以直接进行读写操作。
Tcp 服务端和客户端程序示例如下：

import std.socket.*
import std.time.*
import std.sync.*

var SERVER_PORT: UInt16 = 0

func runTcpServer() {
    try (serverSocket = TcpServerSocket(bindAt: SERVER_PORT)) {
        serverSocket.bind()
        SERVER_PORT = serverSocket.localAddress.port

        try (client = serverSocket.accept()) {
            let buf = Array<Byte>(10, item: 0)
            let count = client.read(buf)

            // 服务端读取到的数据为: [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
            println("Server read ${count} bytes: ${buf}")
        }
    }
}

main(): Int64 {
    let future = spawn {
        runTcpServer()
    }
    sleep(Duration.millisecond * 500)

    try (socket = TcpSocket("127.0.0.1", SERVER_PORT)) {
        socket.connect()
        socket.write(Array<Byte>([1, 2, 3]))
    }

    future.get()

    return 0
}
编译执行上述代码，将打印：

Server read 3 bytes: [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
Udp 编程
Udp 作为一种常见的不可靠传输协议，以 Udp 类型套接字举例，仓颉在不可靠传输场景下的可参考的编程模型如下：

创建套接字，并指定本端绑定地址。
执行绑定。
指定远端地址进行报文发送。
不连接远端地址场景下，可以收取来自不同远端地址的报文，并返回远端地址信息。
Udp 收发报文程序示例如下：

import std.socket.*
import std.time.*
import std.sync.*

let SERVER_PORT: UInt16 = 8080

func runUpdServer() {
    try (serverSocket = UdpSocket(bindAt: SERVER_PORT)) {
        serverSocket.bind()

        let buf = Array<Byte>(3, item: 0)
        let (clientAddr, count) = serverSocket.receiveFrom(buf)
        let sender = clientAddr.hostAddress

        // Server receive 3 bytes: [1, 2, 3] from 127.0.0.1
        println("Server receive ${count} bytes: ${buf} from ${sender}")
    }
}

main(): Int64 {
    let future = spawn {
        runUpdServer()
    }
    sleep(Duration.second)

    try (udpSocket = UdpSocket(bindAt: 0)) {
        udpSocket.sendTimeout = Duration.second * 2
        udpSocket.bind()
        udpSocket.sendTo(
            SocketAddress("127.0.0.1", SERVER_PORT),
            Array<Byte>([1, 2, 3])
        )
    }

    future.get()

    return 0
}
编译执行上述代码，将打印：


Server receive 3 bytes: [1, 2, 3] from 127.0.0.1