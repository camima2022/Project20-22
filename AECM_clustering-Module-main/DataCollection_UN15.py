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
# dns
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
            length = len(packet)
            src_ip0 = packet[28:32]
            src_ip = ".".join(bytetoint(src_ip0))
            src.append(src_ip)

            dst_ip0 = packet[32:36]
            dst_ip = ".".join(bytetoint(dst_ip0))
            dst.append(dst_ip)

            sport0 = packet[36:38]
            src_port = change(sport0)
            sport.append(src_port)

            dport0 = packet[38:40]
            dst_port = change(dport0)
            dport.append(dst_port)

            tcp_flag = packet[25]
            tcp.append(str(tcp_flag))

            raw_data = packet[44:length]

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
# bittorrent and ssh and nfs
def readPcap_ssh(file):
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
            src_ip0 = packet[28:32]
            src_ip = ".".join(bytetoint(src_ip0))
            src.append(src_ip)

            dst_ip0 = packet[32:36]
            dst_ip = ".".join(bytetoint(dst_ip0))
            dst.append(dst_ip)

            sport0 = packet[36:38]
            src_port = change(sport0)
            sport.append(src_port)

            dport0 = packet[38:40]
            dst_port = change(dport0)
            dport.append(dst_port)

            tcp_flag = packet[25]
            tcp.append(str(tcp_flag))
            raw_data = packet[68:length]
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
# ftp , http, imap, pop, smb, smtp  56
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
            src_ip0 = packet[28:32]
            src_ip = ".".join(bytetoint(src_ip0))
            src.append(src_ip)

            dst_ip0 = packet[32:36]
            dst_ip = ".".join(bytetoint(dst_ip0))
            dst.append(dst_ip)

            sport0 = packet[36:38]
            src_port = change(sport0)
            sport.append(src_port)

            dport0 = packet[38:40]
            dst_port = change(dport0)
            dport.append(dst_port)

            tcp_flag = packet[25]
            tcp.append(str(tcp_flag))
            raw_data = packet[56:length]
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

def DataflowRegroup(payload, session, exchange_session):
    index = 1
    data_flow = []
    count = int(len(session) - 1)

    data_flow.append(payload[0])

    while count > 0:
        try:
            if session[index][0:4] == session[index - 1][0:4]:
                data_last = data_flow[-1]
                data_last = data_last + payload[index]
                data_flow[-1] = data_last
            elif session[index][0:4] == exchange_session[index - 1][0:4]:
                data_flow.append(payload[index])
            else:
                data_flow.append(payload[index])
            count -= 1
            index += 1
        except IndexError:
            break
    return data_flow

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

def DataPayloadCollection(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'dns':
                    data, session, exchange_session = readPcap_dns(file=cwd)
                #
                elif class_list[i] == 'ssh' or class_list[i] == 'bittorrent':
                    data, session, exchange_session = readPcap_ssh(file=cwd)

                else:
                    data, session, exchange_session = readPcap_ftp(file=cwd)
                data = DataflowRegroup(data, session, exchange_session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    # print('0-dns:{}\n1-ftp:{}\n2-imap:{}\n3-smtp:{}\n'.format(labels.count(0), labels.count(1), labels.count(2),
    #                                                          labels.count(3)))
    print('0-:{}\n1-:{}\n2-:{}\n3-:{}\n'.format(labels.count(0), labels.count(1), labels.count(2),
                                                              labels.count(3)))

    payload = DataSegementation(payload_data=payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float64") / 255
    # data_numpy = np.asarray(payload, dtype="float64")

    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy

from sklearn.cluster import KMeans, DBSCAN, OPTICS
import numpy as np
import metrics
from sklearn.model_selection import train_test_split
from time import time

def load_dataset_and_clustering():

    # class_list = ['bittorrent', 'imap', 'ssh', 'smtp']

    A = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop', 'smb', 'smtp', 'ssh']

    path = './UN15_20000_version/'
    num_clusters = 4

    for i in range(9):
        for j in range(i+1, 9):
            for k in range(j+1, 9):
                for p in range(k+1, 9):

                    class_list = [A[i], A[j], A[k], A[p]]
                    data, label = DataPayloadCollection(path, class_list)
                    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)

                    x = np.concatenate((train_x, test_x))
                    y = np.concatenate((train_y, test_y))
                    km = KMeans(n_clusters=num_clusters, n_init=20, n_jobs=4)
                    t0 = time()
                    km_pred = km.fit_predict(x)
                    print("i, j, k, p: ", i, j, k, p)
                    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
                          % (metrics.acc(y, km_pred, num_cluster=num_clusters), metrics.nmi(y, km_pred), metrics.ari(y, km_pred)))

                    print('clustering time: ', (time() - t0))
    # # num_clusters = 4
    # from sklearn.metrics import silhouette_score
    # sc = silhouette_score
    #
    # data, label = DataPayloadCollection(path, class_list)
    # train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.25, random_state=0, shuffle=True)
    #
    # x = np.concatenate((train_x, test_x))
    # y = np.concatenate((train_y, test_y))
    # # x = x.reshape((x.shape[0], length, 1))
    # print('dataset shapes', x.shape)
    # for num_clusters in range(4, 11):
    #     km = KMeans(n_clusters=num_clusters, n_init=20, n_jobs=4)
    #     t0 = time()
    #     km_pred = km.fit_predict(x)
    #     print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
    #           % (metrics.acc(y, km_pred, num_cluster=num_clusters), metrics.nmi(y, km_pred), metrics.ari(y, km_pred)))
    #     # print(' ' * 8 + '|==>  sc: %.4f,  nmi: %.4f,  ari: %.4f  <==|'
    #     #       % (sc(x, km_pred), metrics.nmi(y, km_pred), metrics.ari(y, km_pred)))
    #     # print(' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f  <==|'
    #     #       % (metrics.nmi(y, km_pred), metrics.ari(y, km_pred)))
    #     print('clustering time: ', (time() - t0))

# load_dataset_and_clustering()
# path = '/home/hbb/pythonProject/data/temp/'
# class_list = ['dns', 'http', 'imap', 'pop']
# data, label = DataPayloadCollection(path, class_list)
# cwd = '/home/hbb/pythonProject/data/dns/dns.pcap'
# data, session, exchange_session = readPcap_dns(file=cwd)
# data = DataflowRegroup(data, session, exchange_session)
# print(len(data))



