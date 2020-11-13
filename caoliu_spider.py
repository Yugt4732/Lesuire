# -*- coding:UTF-8 -*-
import requests
from bs4 import BeautifulSoup as BS
import os
from Tools.basic_method import *
import re
from tqdm import tqdm
from multiprocessing import Process
import urllib
from lxml import etree

import matplotlib.pyplot as plt
import time
import random

CONCURRENT_REQUESTS = 1
DOWNLOAD_DELAY = 5


class spd():

    #
    # def hub(self, idx):
    #     self.path = '/Users/momo/Yugangtian/Datasets/caoliu/'
    #     maxspan = 100
    #     for i in range(88, maxspan + 1):
    #         if i % 3 != idx:
    #             continue
    #         print('page:' + str(i) + '-----------------------------'+'\n')
    #         url = 'http://t66y.com/thread0806.php?fid=8{}'.format(
    #             ('&search=&page=' + str(i)) if i != 1 else '')
    #
    #         html = self.request(url)
    #         print('html=',html)
    #         # print(html)
    #         all_a = BS(html.text.encode('gbk'), 'lxml')
    #         print("text= ",all_a)
    #         all_a = all_a.find('tbody', style='table-layout:fixed;')
    #         # print(all_a)
    #         all_a = all_a.find_all('a', id=re.compile('.*'),
    #                                 href=re.compile('.*html'))  # , class_='entry-media__wrapper czr__r-i no-centering')
    #         # print(all_a)
    #         for a in all_a:
    #             # a = li.find('a')
    #             # print(a)
    #             new_url = 'http://t66y.com/' + a['href']
    #             # print(new_url)
    #             html1 = self.request(new_url)
    #             # print(html1)
    #             html1 = BS(html1.text, 'lxml')
    #             # print(html1)
    #             body = html1.find('div', class_='t t2', style='border-top:0')
    #             # print(len(body))
    #             # print(body)
    #             if body != None:
    #                 img_all = body.find('div', class_='tpc_content do_not_catch')
    #                 # print(img_all)
    #                 if img_all != None:
    #                     img_all = img_all.find_all('img',)# 'data-link'!=None)
    #                     if img_all !=None:
    #                         # print(img_all)
    #                         # img_all = .find_all('',image-big src = re.compile('.*(jpg|png)'))
    #                         # print(img_all)
    #                         for img in img_all:
    #                             # print(img)
    #                             if img['ess-data'] != None:
    #                                 try:
    #                                     Img = img['ess-data']
    #                                     # print('img1= ',Img)
    #                                     name = Img.split('/')[-1]
    #                                     Img = self.request(Img)
    #                                     # print('img2=',Img)
    #
    #                                     print(self.path + name)
    #                                     f = open(self.path + name, 'ab')
    #                                     f.write(Img.content)
    #                                     f.close()
    #                                 except:
    #                                     pass

    def page(self, start_idx, end_idx, idx, mod):
        self.path = './caoliu/'
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        for i in range(start_idx, end_idx+1):
            if i % mod != idx:
                continue
            path = os.path.join(self.path,str(i))
            if not os.path.exists(path):
                os.mkdir(path)
            print('page:' + str(i) + '-----------------------------'+'\n')
            url = 'http://t66y.com/thread0806.php?fid=8{}'.format(
                ('&search=&page=' + str(i)) if i != 1 else '')

            html = self.request(url)
            if html.status_code != 200:
                print(" return not 200. page ",i)
                continue
            all_a = BS(html.text.encode('iso-8859-1'), 'lxml')
            # print(all_a)
            all_a = all_a.find('tbody', style='table-layout:fixed;')
            all_td = all_a.find_all('td', class_='tal', style='padding-left:8px')
            for td in all_td:
                # 图集操作
                if "寫真" in td.text or "亞洲" in td.text:
                    a = td.find('a', target="_blank")
                    set_link = 'http://t66y.com/' + a['href']
                    # print(set_link)
                    dir_name = a.text
                    dir_name = os.path.join(path,dir_name)
                    img_html = self.request(set_link)
                    # 如果图集链接返回值 ！=200， 跳过
                    print("process:",idx, " : return code:",img_html.status_code)
                    if img_html.status_code != 200:
                        print(" return not 200", set_link)
                        continue
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    else:
                        continue
                    try:
                        img_set = BS(img_html.text.encode('iso-8859-1'), 'lxml')
                        img_set = img_set.find('div', class_='tpc_content do_not_catch')
                        img_set = img_set.find_all('img')
                        for img in img_set:
                            img_link = img["ess-data"]
                            img_name = img["ess-data"].split('/')[-1]
                            if img['ess-data'] != None:
                                try:
                                    Img = self.request(img_link)
                                    if Img.status_code != 200:
                                        continue
                                    save_name = os.path.join(dir_name,img_name)
                                    print("idx:%d "%i, save_name)
                                    # print("idx:%d "%i, Img)
                                    f = open(save_name, 'ab')
                                    f.write(Img.content)
                                    f.close()
                                except:
                                    print("img error.")
                                    pass
                        if not os.listdir(dir_name):
                            os.removedirs(dir_name)
                            print(dir_name," is empty. So rm......")
                        if not os.listdir(dir_name.replace("\\", '')):
                            os.removedirs(dir_name.replace(" \\", ''))
                            print(dir_name.replace("\\", '')," is empty. So rm......")
                    except:
                        print("img_set error.")
                        pass








    def request(self, url):
        time.sleep(random.randint(77, 333)*1.0/600)
        #time.sleep(1)
        user_agents = [
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60',
            'Opera/8.0 (Windows NT 5.1; U; en)',
            'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
            'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2 ',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
            'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0) ',
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 "
            "(KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
            "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 "
            "(KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 "
            "(KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 "
            "(KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 "
            "(KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 "
            "(KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 "
            "(KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 "
            "(KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 "
            "(KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 "
            "(KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
        ]
        randdom_header = random.choice(user_agents)
        headers = {'User-Agent':randdom_header,

        }

        req = requests.get(url=url,
                           headers=headers,
                           # verify=False,
                           # cookies=cookie
                           timeout=1
                           )
        return req




if __name__ == "__main__":
    num_process = 3
    sp = spd()
    url = 'http://t66y.com/thread0806.php?fid=8'
    for process_idx in range(num_process):
        p = Process(target=sp.page,args=(1, 100, process_idx, num_process))
        p.start()
        time.sleep(6)
        # all_process.append(p)
    # sp.page(1, 100)
