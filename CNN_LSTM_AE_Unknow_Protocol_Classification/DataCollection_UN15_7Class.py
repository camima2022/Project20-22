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
# read bittorrent and ssh
def readPcap_bittorrent(file):
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

            # 根据tcp、udp进行不同的选取策略：TCP= 12+16+20, UDP= 12+4+20
            transfer_flag = packet[25]
            tcp_flag = packet[25]
            tcp.append(str(tcp_flag))
            # ip_layer_data = packet[14:26]



                # tcp_layer_data = packet[38:54]
            raw_data = packet[68:length]
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

    return payload_data, session ##
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

            # 根据tcp、udp进行不同的选取策略：TCP= 12+16+20, UDP= 12+4+20
            transfer_flag = packet[23]
            tcp_flag = packet[25]
            tcp.append(str(tcp_flag))
            # ip_layer_data = packet[14:26]
            # udp_layer_data = packet[38:42]
            raw_data = packet[44:length] # pure application layer data
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

            raw_data = packet[60:length]

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
def readPcap_remaining(file):
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

            # 根据tcp、udp进行不同的选取策略：TCP= 12+16+20, UDP= 12+4+20
            transfer_flag = packet[25]
            tcp_flag = packet[25]
            tcp.append(str(tcp_flag))
            # ip_layer_data = packet[14:26]



                # tcp_layer_data = packet[38:54]
            raw_data = packet[56:length]
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
    length = 100
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
"""
def DataPayloadCollection(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'dns':
                    data, session = readPcap_dns(file=cwd)
                elif class_list[i] == 'pop3':
                    data, session = readPcap_pop3(file=cwd)
                elif class_list[i] == 'smb':
                    data, session = readPcap_smb(file=cwd)
                else:
                    data, session = readPcap_other(file=cwd)
                data = DataflowRegroup(payload=data, session=session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-dns:{}\n1-ftp:{}\n2-http:{}\n3-pop3:{}\n4-smb:{}\n5-smtp:{}\n6-ssh:{}\n'.format(labels.count(0), labels.count(1), labels.count(2),
                                                             labels.count(3), labels.count(4), labels.count(5), labels.count(6)))

    payload = DataSegementation(payload_data=payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float64") / 255

    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy
"""
def DataPayloadCollection_9class(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'bittorrent':
                    data, session = readPcap_bittorrent(file=cwd)
                elif class_list[i] == 'dns':
                    data, session = readPcap_dns(file=cwd)
                elif class_list[i] == 'smb':
                    data, session = readPcap_smb(file=cwd)
                else:
                    data, session = readPcap_remaining(file=cwd)
                data = DataflowRegroup(payload=data, session=session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-bittorrent:{}\n1-dns:{}\n2-ftp:{}\n3-http:{}\n4-imap:{}\n5-pop3:{}\n6-smb:{}\n7-smtp:{}\n8-ssh:{}\n'.format(labels.count(0), labels.count(1), labels.count(2),
                                                             labels.count(3), labels.count(4), labels.count(5), labels.count(6), labels.count(7), labels.count(8)))

    payload = DataSegementation(payload_data=payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float64") / 255

    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy
def DataPayloadCollection_7class_v1(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'bittorrent':
                    data, session = readPcap_bittorrent(file=cwd)
                elif class_list[i] == 'dns':
                    data, session = readPcap_dns(file=cwd)
                elif class_list[i] == 'smb':
                    data, session = readPcap_smb(file=cwd)
                else:
                    data, session = readPcap_remaining(file=cwd)
                data = DataflowRegroup(payload=data, session=session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-dns:{}\n1-ftp:{}\n2-http:{}\n3-imap:{}\n4-pop3:{}\n5-smb:{}\n6-smtp:{}\n'.format(labels.count(0), labels.count(1), labels.count(2),
                                                             labels.count(3), labels.count(4), labels.count(5), labels.count(6)))

    payload = DataSegementation(payload_data=payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float64") / 255

    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy
def DataPayloadCollection_7class_v2(path, class_list):
    payload_data = []
    labels = []
    for name in os.listdir(path):
        cwd = path + name
        for i in range(len(class_list)):
            label = []
            if class_list[i] in cwd:
                if class_list[i] == 'bittorrent':
                    data, session = readPcap_bittorrent(file=cwd)
                elif class_list[i] == 'dns':
                    data, session = readPcap_dns(file=cwd)
                elif class_list[i] == 'smb':
                    data, session = readPcap_smb(file=cwd)
                else:
                    data, session = readPcap_remaining(file=cwd)
                data = DataflowRegroup(payload=data, session=session)
                payload_data.extend(data)
                for j in range(len(data)):
                    label.append(i)
                labels.extend(label)
    print('0-bittorrent:{}\n1-dns:{}\n2-ftp:{}\n3-http:{}\n4-imap:{}\n5-smtp:{}\n6-ssh:{}\n'.format(labels.count(0), labels.count(1), labels.count(2),
                                                             labels.count(3), labels.count(4), labels.count(5), labels.count(6)))

    payload = DataSegementation(payload_data=payload_data)
    print("payload_data:", len(payload), "label:", len(labels))
    # 数据流内容特征 - 类标签
    data_numpy = np.asarray(payload, dtype="float64") / 255

    label_numpy = np.asarray(labels)
    return data_numpy, label_numpy
def cluster_viz(latent, labels, save_dir, n_classes=4, fig_name='tsne_viz.png'):
    '''
    :param latent: 2-D array of predicted latent vectors for each sample
    :param labels: vector of true labels
    :param save_dir: string for destination directory
    :param n_classes: integer for number of classes
    :param fig_name: string for figure name
    '''
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    from sklearn.manifold import TSNE
    colors = cm.rainbow(np.linspace(0, 1, n_classes))

    tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
    tsne_embed = tsne.fit_transform(latent)

    fig, ax = plt.subplots(figsize=(16, 10))
    for i, c in zip(range(n_classes), colors):
        ind_class = np.argwhere(labels == i)
        ax.scatter(tsne_embed[ind_class, 0], tsne_embed[ind_class, 1], color=c, label=i, s=5)
    ax.set_title('t-SNE vizualization of latent vectors')
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'{save_dir}/{fig_name}')
    plt.close('all')

# 9 class
#class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'pop', 'smb', 'smtp', 'ssh']
# 8 class
#class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'smb', 'smtp', 'ssh']
# 7 class v1 -- UN_7class_v1
#class_list = ['dns', 'ftp', 'http', 'imap', 'pop', 'smb', 'smtp']
class_list = ['bittorrent', 'dns', 'ftp', 'http', 'imap', 'smtp', 'ssh']
path = "/home/hbb/pythonProject/Protocol_dataset/UN15_7class_v2/"
save_dir = '/home/hbb/pythonProject/Protocol_dataset/Figure'

data, label = DataPayloadCollection_7class_v2(path, class_list)
cluster_viz(data, label, save_dir, n_classes=7, fig_name='tsne_viz_un15_7class_v2.png')