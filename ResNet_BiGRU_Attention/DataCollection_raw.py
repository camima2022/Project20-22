"""
数据预处理
"""
from __future__ import print_function
import struct
import numpy as np
import os
from scipy import misc
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf


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


def readPcap(pcapfile):
    """
    读取pcap文件，获取应用层协议数据
    :param pcapfile: 单一协议的 pcap文件
    :param type: 协议类型标识
    :return: 应用层 payload列表
    """
    fr = open(pcapfile, 'rb')
    string_data = fr.read()
    packet_num = 0
    payload_data = []
    pcap_packet_header = {}
    i = 24
    src = []
    dst = []
    sport = []
    dport = []
    tcp = []
    flag = []
    while (i<len(string_data)):
        src_ip0 = string_data[i + 42:i + 46]
        src_ip = ".".join(bytetoint(src_ip0))
        src.append(src_ip)
        dst_ip0 = string_data[i + 46:i + 50]
        dst_ip = ".".join(bytetoint(dst_ip0))
        dst.append(dst_ip)
        sport0 = string_data[i + 50:i + 52]
        src_port = change(sport0)
        sport.append(src_port)
        dport0 = string_data[i + 52:i + 54]
        dst_port = change(dport0)
        dport.append(dst_port)

        tcp_flag = string_data[i + 39]
        tcp.append(str(tcp_flag))
        pcap_packet_header['len'] = string_data[i + 12:i + 16]
        transferLayerFlag = string_data[i + 39]
        packet_len = struct.unpack('I', pcap_packet_header['len'])[0]

        if transferLayerFlag == int(6):
            raw_data = string_data[i + 70:i + 16 + packet_len]
            payload_data.append(raw_data)
        if transferLayerFlag == int(17):
            raw_data = string_data[i + 58:i + 16 + packet_len]
            payload_data.append(raw_data)
        i = i + packet_len + 16
        packet_num += 1
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
    fr.close()
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
                data_last = data_flow[-1]
                data_joint = data_last + payload[index + 1]
                session[index + 1][-1] = '1'
                data_flow[-1] = data_joint
            count -= 1
            index += 1
        except IndexError:
            break
    # for i in range(len(data_flow)):
    #     label.append(type)
    # return data_flow, label
    return data_flow

# fixed length
def DataSegementation(payload_data):
    """
    数据分割
    :param payload_data:type:list,
    :return:
    """
    length = 256
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

# DataSet DARPA
def DataPayloadCollection(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                data, session = readPcap(pcapfile=cwd)
                if class_list[i] == 'smtp' or class_list[i] == 'http':
                    data = DataflowRegroup(payload=data, session=session)
                # print(class_list[i], len(data))
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-dns:{}\n1-ftp:{}\n2-smtp:{}\n3-http:{}\n'.format(labels.count(0),labels.count(1), labels.count(2), labels.count(3)))
    payload = DataSegementation(payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float") / 255
    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy

# DataSet USTC-TFC2016
def DataCollection_USTC(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                data, session = readPcap(pcapfile=cwd)
                if class_list[i] == 'http':
                    data = DataflowRegroup(payload=data, session=session)
                # print(class_list[i], len(data))
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-BitTorrent:{}\n1-DNS:{}\n2-FTP:{}\n3-HTTP:{}\n4-MySQL:{}\n5-Skype:{}\n6-SMB:{}\n7-WOW:{}\n'.format(labels.count(0),
                                                                                    labels.count(1), labels.count(2), labels.count(3), labels.count(4),
                                                                                    labels.count(5), labels.count(6), labels.count(7)))
    payload = DataSegementation(payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float") / 255
    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy

