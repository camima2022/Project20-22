import numpy as np
import  DataCollection
import pandas as pd
"""
arr = np.zeros((16))
print(arr)
arr[0] = arr[0]+10
print(type(arr[0]))

file_list = ['http.pcap', 'ftp.pcap', 'smtp.pcap', 'dns.pcap']
path = '/home/hbb/protocol_identification/1D_CNN_ResNet/RAW_DATA/'
file_length = np.zeros((16))

for file in file_list:
    file_name = path + file
    payload,_ = DataCollection.readPcap(file_name)
    for single in payload:
        index = np.int(len(single) / 100)
        file_length[index] = file_length[index] + 1
#for i in range(16):
#    print(file_length[i])
print(file_length)

#a = np.array([81415, 21543, 6865, 3639, 5274,  1411,   983,  1234,  1068,   400,  2983,   633,   720,  3106, 35598])
a = np.array([102115, 17210, 8933, 4177, 2239,  1138,   640,  2616,  544,   221,  2486,   425,   553,  1729, 24721])
a_pd = pd.DataFrame(a)
writer = pd.ExcelWriter('a.xlsx')
a_pd.to_excel(writer,'sheet1',float_format='%.4f')
writer.save()
writer.close()
"""

import os
import DataCollection

file_list = os.listdir('./raw_ISCX2012')
print(file_list)
filename_list = ['http.pcap', 'imap.pcap', 'ssh.pcap', 'ftp.pcap', 'smtp.pcap', 'bittorrent.pcap', 'dns.pcap', 'smb.pcap', 'pop3.pcap']
path = '/ResNet_BiGRU_Attention/raw_ISCX2012/'

file_length = np.zeros((15))

for file in filename_list:
    file_name = path + file
    payload,_ = DataCollection.readPcap(file_name)
    for single in payload:
        index = np.int(len(single) / 100)
        file_length[index] = file_length[index] + 1
print(file_length)
a = np.array([102115, 17210, 8933, 4177, 2239,  1138,   640,  2616,  544,   221,  2486,   425,   553,  1729, 24721])
a_pd = pd.DataFrame(a)
writer = pd.ExcelWriter('a.xlsx')
a_pd.to_excel(writer,'sheet1',float_format='%.4f')
writer.save()
writer.close()




