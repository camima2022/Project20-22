from __future__ import print_function
import dpkt
import warnings
import numpy as np
import os

warnings.filterwarnings(action='ignore')
def bytetoint(data_list):
    int_data = []
    for i in range(len(data_list)):
        int_data.append(str(data_list[i]))
    return int_data
def change(data_list):
    data = []
    for i in range(len(data_list)):
        data.append((data_list[i]))
    change_data = str(data[0] * 256 + data[1])
    return change_data

def readPcap_dns(file):
    payload_data = []
    src = []
    dst = []
    sport = []
    dport = []
    tcp = []
    flag = []

    with open(file, 'rb') as f:
        packets = dpkt.pcap.Reader(f)
        for ts, packet in packets:
            # 流量清洗，1,去掉帧头；2，去掉ip、port等信息,3 剩余数据进行拼接
            length = len(packet)
            src_ip0 = packet[26:30]
            src_ip = ".".join(bytetoint(src_ip0))
            src.append(src_ip)

            dst_ip0 = packet[30:34]
            dst_ip = ".".join(bytetoint(dst_ip0))
            dst.append(dst_ip)

            sport0 = packet[34:36]
            src_port = change(sport0)
            sport.append(src_port)

            dport0 = packet[36:38]
            dst_port = change(dport0)
            dport.append(dst_port)

            # 根据tcp、udp进行不同的选取策略：TCP= 12+16+20, UDP= 12+4+20
            transfer_flag = packet[23]
            tcp_flag = packet[23]
            tcp.append(str(tcp_flag))
            # ip_layer_data = packet[14:26]
            # udp_layer_data = packet[38:42]
            raw_data = packet[42:length] # pure application layer data
            # ids dns:raw_data = packet[42:length]

            payload_data.append(raw_data)
            flag.append('0')

    session = []
    for i in range(len(src)):
        s = []
        s.append(src[i])
        s.append(dst[i])
        s.append(sport[i])
        s.append(dport[i])
        s.append(tcp[i])
        s.append(flag[i])
        session.append(s)

    return payload_data, session
def readPcap_http(file):
    payload_data = []
    src = []
    dst = []
    sport = []
    dport = []
    tcp = []
    flag = []

    with open(file, 'rb') as f:
        packets = dpkt.pcap.Reader(f)
        for ts, packet in packets:
            # 流量清洗，1,去掉帧头；2，去掉ip、port等信息,3 剩余数据进行拼接
            length = len(packet)
            src_ip0 = packet[26:30]
            src_ip = ".".join(bytetoint(src_ip0))
            src.append(src_ip)

            dst_ip0 = packet[30:34]
            dst_ip = ".".join(bytetoint(dst_ip0))
            dst.append(dst_ip)

            sport0 = packet[34:36]
            src_port = change(sport0)
            sport.append(src_port)

            dport0 = packet[36:38]
            dst_port = change(dport0)
            dport.append(dst_port)

            # 根据tcp、udp进行不同的选取策略：TCP= 12+16+20, UDP= 12+4+20
            transfer_flag = packet[23]
            tcp_flag = packet[23]
            tcp.append(str(tcp_flag))
            # ip_layer_data = packet[14:26]



                # tcp_layer_data = packet[38:54]
            raw_data = packet[66:length]
            # ids http:raw_data = packet[66:length]
            #    ftp: raw_data = packet[54:length]
            #    smb: raw_data = packet[58:length]
            # raw_data = packet[14:length]


            payload_data.append(raw_data)
            flag.append('0')

    session = []
    for i in range(len(src)):
        s = []
        s.append(src[i])
        s.append(dst[i])
        s.append(sport[i])
        s.append(dport[i])
        s.append(tcp[i])
        s.append(flag[i])
        session.append(s)

    return payload_data, session
def readPcap_ftp(file):
    payload_data = []
    src = []
    dst = []
    sport = []
    dport = []
    tcp = []
    flag = []

    with open(file, 'rb') as f:
        packets = dpkt.pcap.Reader(f)
        for ts, packet in packets:
            # 流量清洗，1,去掉帧头；2，去掉ip、port等信息,3 剩余数据进行拼接
            length = len(packet)
            src_ip0 = packet[26:30]
            src_ip = ".".join(bytetoint(src_ip0))
            src.append(src_ip)

            dst_ip0 = packet[30:34]
            dst_ip = ".".join(bytetoint(dst_ip0))
            dst.append(dst_ip)

            sport0 = packet[34:36]
            src_port = change(sport0)
            sport.append(src_port)

            dport0 = packet[36:38]
            dst_port = change(dport0)
            dport.append(dst_port)

            # 根据tcp、udp进行不同的选取策略：TCP= 12+16+20, UDP= 12+4+20
            transfer_flag = packet[23]
            tcp_flag = packet[23]
            tcp.append(str(tcp_flag))
            # ip_layer_data = packet[14:26]

            raw_data = packet[54:length]

            payload_data.append(raw_data)
            flag.append('0')

    session = []
    for i in range(len(src)):
        s = []
        s.append(src[i])
        s.append(dst[i])
        s.append(sport[i])
        s.append(dport[i])
        s.append(tcp[i])
        s.append(flag[i])
        session.append(s)

    return payload_data, session
def readPcap_smb(file):
    payload_data = []
    src = []
    dst = []
    sport = []
    dport = []
    tcp = []
    flag = []

    with open(file, 'rb') as f:
        packets = dpkt.pcap.Reader(f)
        for ts, packet in packets:
            # 流量清洗，1,去掉帧头；2，去掉ip、port等信息,3 剩余数据进行拼接
            length = len(packet)
            src_ip0 = packet[26:30]
            src_ip = ".".join(bytetoint(src_ip0))
            src.append(src_ip)

            dst_ip0 = packet[30:34]
            dst_ip = ".".join(bytetoint(dst_ip0))
            dst.append(dst_ip)

            sport0 = packet[34:36]
            src_port = change(sport0)
            sport.append(src_port)

            dport0 = packet[36:38]
            dst_port = change(dport0)
            dport.append(dst_port)

            # 根据tcp、udp进行不同的选取策略：TCP= 12+16+20, UDP= 12+4+20
            transfer_flag = packet[23]
            tcp_flag = packet[23]
            tcp.append(str(tcp_flag))
            # ip_layer_data = packet[14:26]

            raw_data = packet[58:length]

            payload_data.append(raw_data)
            flag.append('0')

    session = []
    for i in range(len(src)):
        s = []
        s.append(src[i])
        s.append(dst[i])
        s.append(sport[i])
        s.append(dport[i])
        s.append(tcp[i])
        s.append(flag[i])
        session.append(s)

    return payload_data, session

def DataflowRegroup(payload, session):
    index = 0
    data_flow = []
    count = len(session)
    while count > 0:  # 数据流重组
        try:
            if session[index][0:4] != session[index + 1][0:4]:  # 方向不同，保存元素
                if session[index][-1] == '0':  # 当前数据包未保存
                    session[index][-1] = '1'
                    data_flow.append(payload[index])
                    # session[index + 1][-1] = '1'
                    # data.append(payload[index+1])
                if session[index + 1][-1] == '0':  # 当前数据包的下一个数据包未保存
                    session[index + 1][-1] = '1'
                    data_flow.append(payload[index + 1])
            else:  # 方向相同，拼接元素
                if data_flow:  # 如果pcap文件前两个数据包的session相同 and 此时data_flow不为空
                    data_last = data_flow[-1]
                    data_joint = data_last + payload[index + 1]
                    session[index + 1][-1] = '1'
                    data_flow[-1] = data_joint
                else:  # 如果pcap文件前两个数据包的session相同 and 此时data_flow为空
                    data_last = payload[index]
                    data_joint = data_last + payload[index + 1]
                    # session[index][-1] = '1'
                    session[index][-1] = '1'
                    session[index + 1][-1] = '1'
                    data_flow.append(data_joint)
            count -= 1
            index += 1
        except IndexError:
            break
    # for i in range(len(data_flow)):
    #     label.append(type)
    # return data_flow, label
    return data_flow

def DataSegementation(payload_data):
    """
    数据分割
    :param payload_data:type:list,
    :return:
    """
    length = 144 # 144
    example = np.zeros((len(payload_data), length))
    for i in range(len(payload_data)):
        value = []
        single = payload_data[i]
        for j in range(len(single)):
            value.append((single[j]))
        while len(value) < length:
            value.append(0)
        if len(value) > length:
            value = value[:length]
        example[i] = np.array(value)
    return example

def DataPayloadCollection(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path=path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'dns':
                    data, session = readPcap_dns(file=cwd)
                elif class_list[i] == 'ftp':
                    data, session = readPcap_ftp(file=cwd)
                elif class_list[i] == 'http':
                    data, session = readPcap_http(file=cwd)
                else:
                    data, session = readPcap_smb(file=cwd)
                data = DataflowRegroup(payload=data, session=session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-dns:{}\n1-ftp:{}\n2-http:{}\n3-smb:{}\n'.format(labels.count(0), labels.count(1), labels.count(2),
                                                             labels.count(3)))

    payload = DataSegementation(payload_data=payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float64") / 255

    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy



