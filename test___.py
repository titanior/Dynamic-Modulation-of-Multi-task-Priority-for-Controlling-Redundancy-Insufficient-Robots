# -*- encoding: utf-8 -*-
import xlwt  
 
def txt_xls(filename, xlsname):
    try:
        f = open(filename, 'r', encoding='utf-8')
        xls = xlwt.Workbook()
        sheet = xls.add_sheet('sheet1', cell_overwrite_ok=True)
        x = 0
        while True:
            # 按行循环，读取文本文件
            line = f.readline()
            if not line:
                break
            for i in range(len(line.split('\t'))):
                item = line.split('\t')[i]
                sheet.write(x, i, item)
            x += 1
        f.close()
        xls.save(xlsname)  # 保存xls文件
    except:
        raise
 
if __name__ == "__main__":
    filename = "./log/progress.txt"   #需要转化的文件
    xlsname = "b.xls"     #保存及命名
    txt_xls(filename, xlsname)