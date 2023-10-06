---
title: "Reptile Learning"
date: 2023-09-24
draft: false

tags: ["python"]
categories: ["reptile"]
---

**urllib库（request更方便，在于不用再去构建一个get或者post请求）**

```python
urllib库包含以下几个模块（模块里面是函数）：

urllib.request - 打开和读取 URL。
urllib.error - 包含 urllib.request 抛出的异常。
urllib.parse - 解析 URL。
# urllib.robotparser - 解析 robots.txt 文件。

拓:
getcode() 函数获取'网页状态码'，返回 200 说明网页正常，返回 404 说明网页不存在
get_text()获取除标签以外的'文本内容'。
.attrs获取'标签对象的属性'，返回一个字典。
body = '\n'.join([line.text for line in lines]) # 有意思的一种换行循环加入列表。
title = bs.find('h1').text  # text什么意思,去除标签获取文本。
用类来设置属性。
zip()函数：for i n in zip(is,ns):['用以设置一个双循环']
str.strip([chars]);   参数chars -- 移除字符串'头尾'指定的字符序列。

1.'urllib.request模块'

urllib.request.urlopen(url, data=None, [timeout, ]*, cafile=None, capath=None, cadefault=False, context=None)
url：url 地址。
data：发送到服务器的其他数据对象，默认为 None。
timeout：设置访问超时时间。
cafile 和 capath：cafile 为 CA 证书， capath 为 CA 证书的路径，使用 HTTPS 需要用到。
cadefault：已经被弃用。
context：ssl.SSLContext类型，用来指定 SSL 设置。


urllib.request.Request(url, data=None, headers={}, origin_req_host=None, unverifiable=False, method=None)
url：url 地址。
data：发送到服务器的其他数据对象，默认为 None。
# headers：HTTP 请求的头部信息，字典格式。
origin_req_host：请求的主机地址，IP 或域名。
unverifiable：很少用整个参数，用于设置网页是否需要验证，默认是Fals                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          e。。
# method：请求方法， 如 GET、POST、DELETE、PUT等。


2.'urllib.error'模块
urllib.error 模块为 urllib.request 所引发的异常定义了异常类，基础异常类是 URLError。
urllib.error 包含了两个方法，URLError 和 HTTPError。
URLError 是 OSError 的一个子类，用于处理程序在遇到问题时会引发此异常（或其派生的异常），包含的属性 reason 为引发异常的原因。
HTTPError 是 URLError 的一个子类，用于处理特殊 HTTP 错误例如作为认证请求的时候，包含的属性 code 为 HTTP 的状态码， reason 为引发异常的原因，headers 为导致 HTTPError 的特定 HTTP 请求的 HTTP 响应头

```

**BeautifulSoup**

```python
bs = BeautifulSoup(html.read(), 'html.parser')  # 'html.parser是python3中的解析器,lxml解析器需要特定安装,还有html5lib解析器。			'特点就是可以转换成一个对象'
print(bs.body.h1) # 返回一个网页标题。

```

**几种异常的获取和分类**

```python
1、网页在服务器上不存在或者获取网页时出现错误（HTTP异常）

2、服务器不存在（URL异常）

3.AttributeError：这个错误就是说python找不到对应的对象的属性（一般是没有标签）

from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import URLError


def gettiller(url):
    try:
        html = urlopen(url)
    except URLError:
        return None  # except执行后就不会执行后面的代码
    try:
        bs = BeautifulSoup(html.read(), 'html.parser')
        title = bs.body.h1
    except AttributeError as e:  # 几种错误的分类。
        return None
    return title

title = gettiller('http://www.pythonscraping.com/pages/page1.html')
if title is None:
    print('无法找到页面')
else:
    print(title)

```

**HTML解析**

```python
Beautifulsoup库里面的四个对象：
1.BeautifulSoup对象获取整个文档
2.标签Tag对象函数find与函数find_all获取
3.NavigableString对象表示标签里面的文字。（需要了解函数）
4.comment对象获取注释（需要了解函数）

标签对象.get_text()会清除HTML文档中的所有标签，返回字符串

# find（）与fand_all（）函数：
find(标签，标签属性（{class:''})，递归布尔变量，text='匹配文本内容'，keuwords关键字参数）
find_all() 
[关键字keywords:'id是一种HTML属性'（都可以直接用claas的字典属性获取），class_='green'直接获取具有green颜色属性的标签]

导航树：
1.子标签和父标签以及后代标签（区别）
.descendants后代标签
.children子标签
2.处理兄弟标签
.next_siblings获取所有'兄弟标签'
.previous_siblings当你知道兄弟标签的最后一个标签后可获取前面的'所有标签'（当然得是复数形式）
3.处理父标签
.parent
.parents
例子：print(bs.find('img', {'src': '../img/gifts/img1.jpg'}).parent.previous_sibling.get_text())
4.正则表达式和BeautifulSoup
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
html = urlopen('https://www.pythonscraping.com/pages/page3.html')
bs = BeautifulSoup(html.read(), 'html.parser')
images = bs.find_all('img', dict(src=re.compile('\.\./img/gifts/img.*\.jpg')))
for image in images:
    print(image.attrs['src'])  # 可以获取图片的地址。
5.Lambda表达式
#不太会
```

**编写网络爬虫**

```python
1.网页跳转（'急需改进'）
from urllib.request import urlopen
from bs4 import BeautifulSoup
import random
import datetime
random.seed(datetime.datetime.now())  # 随机数
def getLinks(art):
    html = urlopen('https://en.wikipedia.beta.wmflabs.org{}'.format(art))
    bs = BeautifulSoup(html, 'html.parser')
    return bs.find('div', {'id': 'mw-content-text'}).find_all('a', {'class': 'new'}) # 不可以用class，太死板。
links = getLinks('/wiki/Kyberpunk')
while len(links) > 0:
    newArticle = links[random.randint(0, len(links) - 1)].attrs['href']
    print(newArticle)
    links = getLinks(newArticle)
2.随机数种子
random.seed(datetime.datetime.now()) # 不知道何用？
3.爬取网站('还是不知道错在哪？')
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
page = set()
def getLinks(art):
    global page
    html = urlopen('https://en.wikipedia.beta.wmflabs.org{}'.format(art))
    bs = BeautifulSoup(html.read(), 'html.parser')
    for link in bs.find_all('a', href=re.compile('^(/wiki/)')):
        if 'href' in link.attrs:
            if link.attrs['href'] not in page:
                newPage = link.attrs['href']
                print(newPage)
                page.add(newPage)
                getLinks(newPage) 
getLinks('')
3.请求头的设置
import urllib.request '要先导入urllib.request模块'
header = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62'}
 resquest = urllib.request.Request('https://en.wikipedia.beta.wmflabs.org' ,headers=header)
 html = urlopen(resquest)
 bs = BeautifulSoup(html.read(), 'html.parser')
```

**收集网站数据**

```python
1.from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
pages = set()
def getLink(art):
    global pages
    html = urlopen('https://en.wikipedia.beta.wmflabs.org{}'.format(art))
    bs = BeautifulSoup(html.read(), 'html.parser')
    try:
        print(bs.h1.get_text())		# 这种多行检测容易失去一些数据。 
        print(bs.find(id='mw-content-text').find_all('p'))
        print(bs.find(id='ca-edit').find('span').find('a').attrs['href'])
    except AttributeError:	# 即使报错也也会进行下一步。
        print('页面缺少了一些属性.')
    for link in bs.find_all('a', href=re.compile('^(/wiki/)')):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                newpPage = link.attrs['href']
                print('-' * 20)
                print(newpPage)
                pages.add(newpPage)
                getLink(newpPage)
getLink('')
难点：各种标签的规律（花时间去了解一下标签的的东西）。


2.'startswith()'方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False。如果参数 beg 和 end 指定值，则在指定范围内检查。
语法
startswith()方法语法：
str.startswith(str, beg=0,end=len(string));
参数
str -- 检测的字符串。
strbeg -- 可选参数用于设置字符串检测的起始位置。
strend -- 可选参数用于设置字符串检测的结束位置。
返回值
如果检测到字符串则返回True，否则返回False。
```

**在互联网上抓取**

```python
from urllib.request import urlopen
from urllib.parse import urlparse	# 用于解析
from bs4 import BeautifulSoup
import re
import datetime
import random

pages = set()
random.seed(datetime.datetime.now())

def getInUrl(bs, includeUrl): # 搜集内链
    includeUrl = '{}://{}'.format(urlparse(includeUrl).scheme, urlparse(includeUrl).netloc)
    inLinks = []
    for link in bs.find_all('a', href=re.compile('^(/|.*' + includeUrl + ')')):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in inLinks:
                if link.attrs['href'].startswith('/'):
                    inLinks.append(includeUrl + link.attrs['href'])
                else:
                    inLinks.append(link.attrs['href'])
    return inLinks

def getExitUrl(bs, exUrl):	# 搜集外链
    exLinks = []
    for link in bs.find_all('a', href=re.compile('^(http|www)((?!' + exUrl + ').)*$')):  # 正则表达式。
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in exLinks:
                exLinks.append(link.attrs['href'])
    return exLinks
    
def getRandomExternalLinks(startingPage):  # 随机获取外部链接。（开始页面)
    html = urlopen(startingPage)
    bs = BeautifulSoup(html.read(), 'html.parser')
    exLinks = getExitUrl(bs, urlparse(startingPage).netloc)  # 获取域名,获取外链。
    if len(exLinks) == 0:
        print('不存在外部链接（一般不可能)')
        domain = '{}://{}'.format(urlparse(startingPage).scheme, urlparse(startingPage).netloc)
        internalLink = getInUrl(bs, domain)  # 就去寻求另一个内链。
        return getRandomExternalLinks(internalLink[random.randint(0, len(internalLink) - 1)])
    else:
        return exLinks[random.randint(0, len(exLinks))]  # 随机返回一个外链
        
def followExternalOnly(startingSite):	# 实现网站跳转。
    exLink = getRandomExternalLinks(startingSite)
    print('随机获取的外部链接是{}'.format(exLink))
    followExternalOnly(exLink)
followExternalOnly('http://oreilly.com')

1.'urlparse()'函数可以将 URL 解析成 ParseResult对象。对象中包含了六个元素，分别为：
from urllib.parse import urlparse
协议（.scheme） 
域名（.netloc） 
路径（.path） 
路径参数（.params） 
查询参数（.query） 
片段（.fragment）
2.'select()'函数[相当于fand_all的用法]
我们在写 CSS 时，标签名不加任何修饰，类名（class="className"引号内即为类名）前加点，id名（id="idName"引号前即为id名）前加 #，在这里我们也可以利用类似的方法来筛选元素，用到的方法是 bs.select()，返回类型是 list

3.以下是request.exceptions下的各种异常错误：
'RequestException'

4.Python replace() 方法用于把字符串中的 old（旧字符串） 替换成 new(新字符串)，如果指定第三个参数max，则替换不超过 max 次。
语法：
str.replace(old, new[, max])

5.os.path.exists()就是判断括号里的文件是否存在的意思【返回布尔值】
os.makedirs递归创建目录（无返回值）

6.获取图片
import requests
import re
import time
import os
url = 'https://www.yunxing.club/4380/.html'
header = {''}
response = requests.get(url, headers=header)
dir_name = re.findall('<h1 class="article-title">(.*?)</h1>', response.text)[0]	# 文件夹名称
html = response.text
urls = re.findall('<img loading=".*?" src="(.*?)" alt="" class=".*?" width=".*?" height=".*?" srcset=".*?" />', html)
if not os.path.exists(dir_name):
    os.mkdir(dir_name) # 创建一个文件夹
for url in urls:
    time.sleep(0.5)
    file_name = url.split('/')[-1]	# 获取图片名称
    response1 = requests.get(url, headers=header)
    with open(dir_name + '/' + file_name, 'wb') as f:	# 逐个加入文件进入文件夹
        f.write(response1.content)

7.try  finally :
finally在return前执行，在finally的操作，不会改变已经确定的return的值， finally不能加return语句。出现异常，先找是否有处理器可以处理这个异常有处理器可以处理这个异常，再finally。 
```

**lxml库**

```python
# etree函数：
1.etree.HTML()接受resquest.text返回的字符串并转换成HTML以便xpath解析

2.etree.parse()接受一个文本路径，可以直接解析【默认解析器是Xpath，可以设置】

3.etree.tostring(文本，encoding='').docode()编码与解码。【用以获取可读的数据】


```

**数据清洗以及自然语言处理**

```python
1.马尔可夫链：# 【实质就是对数据的处理，对字典的灵活运用】
import requests
from random import randint

def A(wordList):
    sum = 0
    for word, value in wordList.items(): 
        sum += value
    return sum

def B(wordList):	# 神奇的创造句子的方法。
    randIndex = randint(1, A(wordList))
    for word, value in wordList.items():
        randIndex -= value
        if randIndex <= 0:  # 随机事件，只有数值大的才有可能小于零。
            return word

def C(text):
    text = text.replace('\n', '')
    text = text.replace('"', '')
    punctuation = [',', '.', ';', ':']
    for symbol in punctuation:
        text = text.replace(symbol, '{}'.format(symbol))
    words = text.split(' ')
    words = [word for word in words if word != '']
    print(words)
    wordDict = {}
    for i in range(1, len(words)):  # 蒙逼？
        if words[i - 1] not in wordDict:
            wordDict[words[i - 1]] = {}
        if words[i] not in wordDict[words[i - 1]]:
            wordDict[words[i - 1]][words[i]] = 0
        wordDict[words[i - 1]][words[i]] += 1
    return wordDict

text = str(requests.get('http://pythonscraping.com/files/inaugurationSpeech.txt').text)
wordDict = C(text)
length = 100
chain = ['I']  # 随机输了一个单词。
for i in range(0, length):
    newWord = B(wordDict[chain[-1]]) 
    chain.append(newWord)
print(' '.join(chain))


2.数据标准化：


3.OpenRefine数据处理应用

```

**NLTK自然语言工具包**

对文本的各种特点进行分析的工具。



**post请求**

```python
1.提交数据时要保证变量名称与数据名称一致。

2.Session()函数可以自动检测cookie,不用时刻跟踪cookie。
import requests

session = requests.Session()

params = {'username': 'Ryan', 'password': 'password'}
r = session.post('https://pythonscraping.com/pages/cookies/welcome.php', params)	# session函数会自动记录cookie
print(r.cookies.get_dict())
a = session.get('http://pythonscraping.com/pages/cookies/profile.php')  # 不需要再声明cookie
print(a.text)


3.HTTP基本接入认证。
import requests
import request.auth import AuthBase	   #这个函数有什么作用？
from requests.auth import HTTPBasicAuth

auth = HTTPBasicAuth('ryan', 'password')
r = requests.post('https://pythonscraping.com/pages/auth/login.php', auth=auth)
print(r.text)

```

