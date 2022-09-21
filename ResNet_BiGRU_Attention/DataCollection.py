from __future__ import print_function
import dpkt
import warnings
import numpy as np
import os
from tensorflow.keras.preprocessing import sequence
warnings.filterwarnings(action='ignore')

"""
path = '/home/hbb/ResNet_BiGRU_Attention/ISCX2012/'
class_list = ['BitTorrent','DNS','FTP','HTTP','IMAP','POP3','SMB','SMTP','SSH']
"""

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

            if transfer_flag == int(6):     # TCP
                tcp_layer_data = packet[38:54]
                raw_data = packet[54:length]

            if transfer_flag == int(17):    # UDP
                udp_layer_data = packet[38:42]
                raw_data = packet[42:length]
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

    return payload_data , session
def readPcap_smtp(file):  # smtp协议的应用层载荷是从第67位开始的，和其他的基于TCP的应用层协议不一样
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

            if transfer_flag == int(6):
                tcp_layer_data = packet[38:54]
                raw_data = packet[66:length]  # smtp协议的应用层载荷是从第67位开始的，和其他的基于TCP的应用层协议不一样

            if transfer_flag == int(17):  # UDP
                udp_layer_data = packet[38:42]
                raw_data = packet[42:length]
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

def DataSegementation(payload_data):
    """
    数据分割
    :param payload_data:type:list,
    :return:
    """
    length = 784
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

def DataPayloadCollection(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        #print(name)
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                data, session = readPcap(file=cwd)
                #if class_list[i] == 'http':
                #    data = DataflowRegroup(payload=data, session=session)
                # print(class_list[i], len(data))
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-dns:{}\n1-ftp:{}\n2-http:{}\n3-smtp:{}\n'.format(labels.count(0),labels.count(1), labels.count(2), labels.count(3)))
    payload = DataSegementation(payload_data)
    #payload = payload_data
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float") / 255
    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy

def DataPayloadCollection_ISCX(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'smtp':
                    data, session = readPcap_smtp(file=cwd)
                else:
                    data, session = readPcap(file=cwd)
                #data ,session = readPcap(file=cwd)
                data= DataflowRegroup(payload=data,session=session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-bittorrent:{}\n1-dns:{}\n2-ftp:{}\n3-http:{}\n4-imap:{}\n5-pop3:{}\n6-smb:{}\n7-smtp:{}\n8-ssh:{}\n'.format(labels.count(0),
                                                            labels.count(1), labels.count(2), labels.count(3),labels.count(4), labels.count(5), labels.count(6),
                                                                                                                  labels.count(7), labels.count(8)))
    #payload = DataSegementation(payload_data)
    payload = DataSegementation(payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float") / 255
    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy




"""

    
    

data_path  = '/home/hbb/ResNet_BiGRU_Attention/ISCX2012/'
result_path = '/home/hbb/ResNet_BiGRU_Attention/result/'
class_list = ['BitTorrent','DNS','FTP','HTTP','IMAP','POP3','SMB','SMTP','SSH']


data , label=DataPayloadCollection(path=data_path,class_list = class_list)

file = '/home/hbb/ResNet_BiGRU_Attention/data2/dns.pcap'
raw_data , raw_session = readPcap(file)
data = raw_data[0:10]
session = raw_session[0:10]
for i in range(len(data)):
    print(data[i])

print(session)
datas = DataflowRegroup(payload=data,session=session)
for j in range(len(datas)):
    print(datas[j])
"""
"""

file = '/home/hbb/ResNet_BiGRU_Attention/data2/dns.pcap'
payload_data , _= readPcap(file)
raw_data = payload_data[0:5]
for i in range(5):
    print(len(raw_data[i]))

data = DataSegementation(raw_data)
for i in range(5):
    print(len(data[i]))
print('\n')
print(raw_data[0])
print(data[0])
"""




