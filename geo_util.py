# -*- coding: utf-8 -*-

import pandas as pd
import requests
import urllib.error
from io import BytesIO
import numpy as np
from PIL import Image


class MapTile():
    """ 座標・タイルの種類をもとに地理院タイルを取得して保持するクラス """

    def __init__(self, from_tile, to_tile=None, zoom=None, tile_type="std", ext="png"):
        """ Constructor
        Input
        -----
          from_tile : 取得したい領域内の左上のタイル座標[z, x, y]
          to_tile   : 取得したい領域内の右下のタイル座標[z, x, y] : 単一タイルの時省略可
          zoom      : ズームレベル(from_tile, to_tileのzと異なるズームレベルで取得したければ)
          tile_type : 取得したいタイルの種類
          ext       : 取得したいタイルの拡張子
        Attribute
        ---------
          data  : 取得した地理院タイル(Numpy Array)
          shape : self.dataのshape
        """
        self.from_tile = from_tile
        self.to_tile = to_tile
        self.zoom = zoom
        self.tile_type = tile_type
        self.ext = ext

        # zoomを省略するとfrom_tileと同じになる
        if zoom == None:
            zoom =from_tile[0]

        # to_tileを省略するとfrom_tileと同じになる
        if to_tile == None:
            to_tile = from_tile

        # 地理院タイルのズームレベルの限界
        MAX_ZOOM = 18
        assert zoom <= MAX_ZOOM

        # 指定された領域・ズームレベルおける最も左上のタイル座標
        x1 = from_tile[1] * 2**(zoom - from_tile[0])
        y1 = from_tile[2] * 2**(zoom - from_tile[0])
        # 指定された領域・ズームレベルにおける最も右下のタイル座標
        x2 = (to_tile[1] + 1) * 2**(zoom - to_tile[0]) - 1 # すぐ右下を考え、それから-1
        y2 = (to_tile[2] + 1) * 2**(zoom - to_tile[0]) - 1 # すぐ右下を考え、それから-1

        # 左上〜右下すべてのタイルの座標を計算
        tile_list = []
        for j in range(y1, y2+1):
            tile_row = []
            for i in range(x1, x2+1):
                tile_row.append([zoom, i, j])
            tile_list.append(tile_row)

        # タイルをダウンロードしてself.dataにセット
        full_tile = []
        for tile_row in tile_list:
            tr = []
            for tile in tile_row:
                tile_url = "http://cyberjapandata.gsi.go.jp/xyz/{tile_type}/{z}/{x}/{y}.{ext}"\
                           .format(tile_type=tile_type, z=tile[0], x=tile[1], y=tile[2], ext=ext)
                if ext in ["png", "jpg"]:
                    # 画像の場合
                    try:
                        response = requests.get(tile_url)
                        img_arr = np.array(Image.open(BytesIO(response.content)))
                    except urllib.error.HTTPError:
                        img_arr = np.zeros((256, 256, 3))
                    tr.append(img_arr)
                elif ext == "txt":
                    # 標高csvの場合
                    try:
                        df = pd.read_csv(tile_url, header=None).replace("e", 0)  # 海: "e" -> 0
                        csv_arr = df.values.astype(np.float)  # numpy array
                    except urllib.error.HTTPError:
                        csv_arr = np.zeros((256, 256))
                    tr.append(csv_arr)
                else:
                    raise ValueError("only {.jpg, .png, .txt} supported")
            tr = np.array(tr)
            full_tile.append(np.hstack(tr))
        self.data = np.vstack(np.array(full_tile))
        self.shape = self.data.shape


    def grad(self, x, y):
        """ 指定された地点での勾配を計算する
        Input
        -----
          x, y: 勾配を計算したい地点の座標
        Output
        -----
          grad_x, grad_y: 勾配のx, y成分
        """
        fx, fy = int(x), int(y) # floor
        dx, dy = x-int(x), y-int(y) # decimal part
        if (dx + dy) < 1:
            grad_x = self.data[fy, fx+1] - self.data[fy, fx]
            grad_y = self.data[fy+1, fx] - self.data[fy, fx]
        else:
            grad_x = self.data[fy+1, fx+1] - self.data[fy+1, fx]
            grad_y = self.data[fy+1, fx+1] - self.data[fy, fx+1]
        return grad_x, grad_y
    
    def grad_norm(self, x, y):
        """ 指定された地点での勾配の大きさを計算する """
        gx, gy = self.grad(x, y)
        return np.sqrt(gx**2 + gy**2)

    def grad_angle(self, x, y):
        """ 指定された地点での勾配の角度を計算する """
        gx, gy = self.grad(x, y)
        return np.arctan2(gy, gx)
