import os
import numpy as np
import DataCollection_IDS
import pandas as pd

path = '/home/hbb/pythonProject/CNN_LSTM_AE_Unknow_Protocol_Classification/IDS17_v3/'
class_list = ['dns.pcap', 'ftp.pcap', 'http.pcap', 'smb.pcap']
data = []
for name in class_list:
    filename = path + name
    if name == 'dns.pcap':
        payload, session = DataCollection_IDS.readPcap_dns(filename)
    elif name == 'ftp.pcap':
        payload, session = DataCollection_IDS.readPcap_ftp(filename)
    elif name == 'http.pcap':
        payload, session = DataCollection_IDS.readPcap_http(filename)
    else:
        payload, session = DataCollection_IDS.readPcap_smb(filename)
    payload = DataCollection_IDS.DataflowRegroup(payload, session)
    data.extend(payload)
file_length = np.zeros((16))
for single in data:
        index = np.int(len(single) / 100)
        if index > 15:
            file_length[15] = file_length[15] + 1
        else:
            file_length[index] = file_length[index] + 1
sum = 0
for i in range(16):
    print(file_length[i])
    sum += file_length[i]
print(sum)
a = np.array([18032, 7195, 4538, 5846, 2746, 1146,  1451,  434,  281,   191,  1697,   150,   179,  149, 178, 1301])
a = a / 45514
print(a)
a_pd = pd.DataFrame(a)
writer = pd.ExcelWriter('a.xlsx')
a_pd.to_excel(writer,'sheet1',float_format='%.4f')
writer.save()
