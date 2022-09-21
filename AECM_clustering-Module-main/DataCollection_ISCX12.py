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
    data =[]
    for i in range(len(data_list)):
        data.append((data_list[i]))
    change_data = str(data[0] * 256 + data[1])
    return change_data

def readPcap(file):
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

            transfer_flag = packet[23]
            tcp_flag = packet[23]
            tcp.append(str(tcp_flag))

            if transfer_flag == int(6):     # TCP bit http imap ftp pop          ssh smtp:66
                raw_data = packet[54:length]

            if transfer_flag == int(17):    # UDP dns: 42
                raw_data = packet[42:length]
            payload_data.append(raw_data)
            flag.append('0')

    session = []
    exchange_session = []
    for i in range(len(src)):
        s = []

        s.append(src[i])
        s.append(dst[i])
        s.append(sport[i])
        s.append(dport[i])
        s.append(tcp[i])
        s.append(flag[i])
        session.append(s)

        exchange_s = []
        exchange_s.append(dst[i])
        exchange_s.append(src[i])
        exchange_s.append(dport[i])
        exchange_s.append(sport[i])
        exchange_s.append(tcp[i])
        exchange_s.append(flag[i])
        exchange_session.append(exchange_s)

    return payload_data, session, exchange_session

# ssh smtp
def readPcap_remaining(file):  # smtp协议的应用层载荷是从第67位开始的，和其他的基于TCP的应用层协议不一样
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

            transfer_flag = packet[23]
            tcp_flag = packet[23]
            tcp.append(str(tcp_flag))

            if transfer_flag == int(6):
                raw_data = packet[66:length]  # smtp协议的应用层载荷是从第67位开始的，和其他的基于TCP的应用层协议不一样

            if transfer_flag == int(17):  # UDP
                raw_data = packet[42:length]
            payload_data.append(raw_data)
            flag.append('0')

    session = []
    exchange_session = []
    for i in range(len(src)):
        s = []

        s.append(src[i])
        s.append(dst[i])
        s.append(sport[i])
        s.append(dport[i])
        s.append(tcp[i])
        s.append(flag[i])
        session.append(s)

        exchange_s = []
        exchange_s.append(dst[i])
        exchange_s.append(src[i])
        exchange_s.append(dport[i])
        exchange_s.append(sport[i])
        exchange_s.append(tcp[i])
        exchange_s.append(flag[i])
        exchange_session.append(exchange_s)

    return payload_data, session, exchange_session

def DataSegementation(payload_data):
    """
    数据分割
    :param payload_data:type:list,
    :return:
    """
    length = 128
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
def DataflowRegroup(payload, session, exchange_session):
    index = 1
    data_flow = []
    count = int(len(session) - 1)

    data_flow.append(payload[0])

    while count > 0:
        try:
            if session[index][0:4] == session[index - 1][0:4]:  # single direction flow
                data_last = data_flow[-1]
                data_last = data_last + payload[index]
                data_flow[-1] = data_last
            # elif session[index][0:4] == exchange_session[index - 1][0:4]:
            #     data_flow.append(payload[index])
            else:
                data_flow.append(payload[index])
            count -= 1
            index += 1
        except IndexError:
            break
    return data_flow

def DataPayloadCollection(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'smtp' or class_list == 'ssh':
                    data, session, exchange_session = readPcap_remaining(file=cwd)
                else:
                    data, session, exchange_session = readPcap(file=cwd)
                #data ,session = readPcap(file=cwd)
                data = DataflowRegroup(data, session, exchange_session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-bittorrent:{}\n1-dns:{}\n2-ftp:{}\n3-http:{}\n4-imap:{}\n5-pop:{}\n6-smb:{}\n7-smtp:{}\n8-ssh:{}\n'.format(labels.count(0),
                                                            labels.count(1), labels.count(2), labels.count(3),labels.count(4), labels.count(5), labels.count(6),
                                                                                                                  labels.count(7), labels.count(8)))
    #payload = DataSegementation(payload_data)
    payload = DataSegementation(payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float") / 255
    # data_numpy = np.asarray(payload, dtype="float")
    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy

def DataPayloadCollection_4class(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'smtp' or class_list == 'ssh':
                # if class_list == 'ssh':
                    data, session, exchange_session = readPcap_remaining(file=cwd)
                else:
                    data, session, exchange_session = readPcap(file=cwd)

                data = DataflowRegroup(data, session, exchange_session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-:{}\n1-:{}\n2-:{}\n3-:{}\n'.format(labels.count(0),
                                                            labels.count(1), labels.count(2), labels.count(3)))

    payload = DataSegementation(payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float") / 255

    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy







