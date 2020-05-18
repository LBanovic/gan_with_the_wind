from mega import Mega

import sys

resolution = int(sys.argv[1])
mega = Mega()
m = mega.login('luka.banovich@gmail.com', 'd!W3P2!B2G')
for res in range(2, resolution+1):
    tfrecord = f'tfrecords-r0{res}.tfrecords'
    file = m.find(tfrecord)
    print(f'Downloading {tfrecord}')
    m.download(file)