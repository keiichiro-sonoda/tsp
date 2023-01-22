# 久しぶりの windows 環境
# 巡回セールスマン問題を解きたい
# 並び替えの交叉は01のときと異なるのでその練習をしたい
# 円順列, じゅず順列等は考えずに等価な個体も別物として扱う
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import time
import json

# シード設定
SEED = 255
SEED = int(time.time() * 1000) & 0xffffffff
# print(hex(SEED))

rd.seed(SEED)
np.random.seed(SEED)

# 遺伝子長
# つまり拠点の数
# 4以上
LENGTH = 100

# Traveling Salesman Problem


class TSP():
    # 世代ごとの個体数
    POPULATION = 50
    # トーナメントサイズ
    TOURN_SIZE = 2
    # 突然変異率
    MTN_RATE = 0.5
    # エリート数
    ELITE_NUM = 1
    # 災害率
    DISASTAR_RATE = 0

    # 経路を求めるための座標を与える
    # numpy 配列を与える
    def __init__(self, coordinates):
        self.coordinates = coordinates  # 引数の座標配列をクラス内変数で保持 (解く問題)
        self.makeDistTable()  # 距離テーブルを作る (numpy配列)
        self.makeFirstGene()  # 最初の世代を作る
        self.CHILD_NUM = self.POPULATION - self.ELITE_NUM  # エリートを除いた子供の数を計算しておく
        self.LOG = []  # 初期世代から, 最短距離を格納していく

    # 循環交叉
    # 親を2つ与える
    # 子も2つタプルで返す
    def cyclicCrossover(self, p1, p2):
        # まず子供をコピーしておく
        c1 = p1.copy()
        c2 = p2.copy()
        # ランダムな添え字を一つ選ぶ
        i = rd.randint(0, LENGTH - 1)
        # 最初の値
        st = p1[i]
        # 今見ている値(次?)
        now = p2[i]
        # 固定する添え字のリスト
        fixed = [i]
        # 最初の値に戻るまでくり返し
        while now != st:
            # 親2と同じ値がある親1の添え字を入手
            i = p1.index(now)
            # リストに追加
            fixed.append(i)
            # 次の値を代入
            now = p2[i]
        # 相互の親から引継ぎ
        for i in fixed:
            c1[i] = p2[i]
            c2[i] = p1[i]
        return (c1, c2)

    # 部分写像交叉
    # PMX: Partially-mapped crossover
    def partMapCrossover(self, p1, p2):
        # リストを初期化
        c1 = [-1] * LENGTH
        c2 = c1.copy()
        # 切断点を決定
        cut1, cut2 = self.getTwoCutPoint()
        if cut1 <= cut2:
            stop1 = cut1 + LENGTH
            stop2 = cut2
        else:
            stop1 = cut1
            stop2 = cut2 + LENGTH
        # 入れ替え辞書
        swap_dict1 = {}
        # 切り取った部分を入れ替え
        for i in range(cut1, stop2):
            ind = i % LENGTH
            c2[ind] = p1[ind]
            c1[ind] = p2[ind]
            swap_dict1[p2[ind]] = p1[ind]
        # 辞書の作り直し
        self.remakeSwapDict(swap_dict1)
        # キーと要素の入れ替え
        swap_dict2 = self.inverseDict(swap_dict1)
        # 衝突しない部分はそのまま
        # 衝突する場合は辞書を見て入れ替え
        for i in range(cut2, stop1):
            ind = i % LENGTH
            if p1[ind] in c1:
                c1[ind] = swap_dict1[p1[ind]]
            else:
                c1[ind] = p1[ind]
            if p2[ind] in c2:
                c2[ind] = swap_dict2[p2[ind]]
            else:
                c2[ind] = p2[ind]
        return c1, c2

    # 順序交叉
    def orderCrossOver(self, p1, p2):
        cut1, cut2 = self.getTwoCutPoint()
        #print(cut1, cut2)
        if cut1 <= cut2:  # 切断点に挟まれている部分をコピー
            c1_m = p1[cut1:cut2]
            c2_m = p2[cut1:cut2]
            c1_o = [i for i in p2 if i not in c1_m]
            c2_o = [i for i in p1 if i not in c2_m]
            c1 = c1_o[:cut1] + c1_m + c1_o[cut1:]
            c2 = c2_o[:cut1] + c2_m + c2_o[cut1:]
        else:  # 切断点に挟まれていない部分をコピー
            c1_o = p1[:cut2] + p1[cut1:]
            c2_o = p2[:cut2] + p2[cut1:]
            c1_m = [i for i in p2 if i not in c1_o]
            c2_m = [i for i in p1 if i not in c2_o]
            c1 = c1_o[:cut2] + c1_m + c1_o[cut2:]
            c2 = c2_o[:cut2] + c2_m + c2_o[cut2:]
        return c1, c2

    # 交叉のテスト用関数
    def crossoverTest(self):
        p1 = self.makeRandomPath()
        p2 = self.makeRandomPath()
        print("親1:", p1)
        print("親2:", p2)
        c1, c2 = self.orderCrossOver(p1, p2)
        print("子1:", c1)
        print("子2:", c2)

    # 2つの切断点を返す関数
    # 必ず切断点どうしは2以上離れるようにする
    def getTwoCutPoint(self):
        points = [i for i in range(LENGTH)]  # 全ての選択肢
        cut1 = rd.choice(points)  # 切断点1をランダムに決定
        # 選択肢から切断点とその両隣を除去
        # 例えば左端の切断点1に選ばれた場合, 右端も隣判定となり選択肢から除外
        for i in range(cut1 - 1, cut1 + 2):
            points.remove(i % LENGTH)
        cut2 = rd.choice(points)  # 切断点2
        return cut1, cut2

    # 入れ替え用辞書を作り直す
    # 入れ替えた数値が更に入れ替えられていたら, その値に更新する
    def remakeSwapDict(self, dic):
        # 消去するキー
        rem_val = []
        for k, v in dic.items():
            # 値が辞書のキーになっている間くり返し
            # 交換した数字で閉じている場合は抜ける
            while (v in dic) and (v not in rem_val):
                rem_val.append(v)
                v = dic[v]
            # 辞書になければその値のままか, 更新
            dic[k] = v
        # 消去
        for v in rem_val:
            dic.pop(v)

    # 要素とキーが逆になった辞書を返す
    # 1対1であることを前提とする
    def inverseDict(self, dic):
        # 辞書内包表記は初めてかも
        return {v: k for k, v in dic.items()}

    # トーナメント選択
    def tournament(self):
        # 順位からランダムに一定数選択し, 最も順位が高いものを選ぶ
        return min(rd.sample(range(self.POPULATION), self.TOURN_SIZE))

    # 複数要素をトーナメント選択(非独立)
    # トーナメントサイズは維持
    # トーナメントサイズ + 選択数が個体数を超えないように注意
    def tournamentMult(self, num):
        selected = []
        ranks = list(range(self.POPULATION))
        # 選択数だけくり返し
        for i in range(num):
            # 各抽選
            winner = min(rd.sample(ranks, self.TOURN_SIZE))
            # 選ばれたランクは除去
            ranks.remove(winner)
            # 返り値のリストに追加
            selected.append(winner)
        return selected

    # 任意の2点を入れ替える突然変異
    def swapTwoMut(self, path):
        # 2つの添え字をランダムに選ぶ
        a, b = rd.sample(range(LENGTH), 2)
        tmp = path[a]
        path[a] = path[b]
        path[b] = tmp

    # ずらす突然変異
    # 隣どうしの交換ならswapと変わらない
    def shiftMut(self, path):
        # 2点選択
        a, b = self.getTwoCutPoint()
        # ループのために b を大きくする
        if b < a:
            b += LENGTH
        # 移動させる点
        tmp = path[a]
        for i in range(a + 1, b + 1):
            ind = i % LENGTH
            # 左にずらす
            # -1が末尾を示す性質を利用
            path[ind - 1] = path[ind]
        path[b % LENGTH] = tmp

    # 突然変異
    # 幾つか手法があるため, 確率的に選択する
    def mutation(self, path):
        if rd.randint(0, 1):  # 入れ替え
            self.swapTwoMut(path)
        else:  # ずらし
            self.shiftMut(path)

    # 1位以外の個体を乱数にする
    def disaster(self):
        self.generation[1:] = [self.makeRandomPath()
                               for i in range(self.POPULATION - 1)]

    # 世代を進める (適応度評価とソートは完了していること前提)
    def advGene(self):
        # 災害発生
        if rd.random() < self.DISASTAR_RATE:
            self.disaster()
            return
        # 空リストスタート
        next_gene = []
        # 上限に達するまで交叉
        while len(next_gene) < self.CHILD_NUM:
            # 親の添え字(ランク)を重複無しで選択
            p_indices = self.tournamentMult(2)
            mother = self.generation[p_indices[0]]
            father = self.generation[p_indices[1]]
            # 交叉方法も等確率で選ぶ
            rn = rd.random()
            if rn < 0.33:  # 循環交叉
                child1, child2 = self.cyclicCrossover(mother, father)
            elif rn < 0.66:  # 部分写像交叉
                child1, child2 = self.partMapCrossover(mother, father)
            else:  # 順序交叉
                child1, child2 = self.orderCrossOver(mother, father)
            # 一定確率で突然変異
            if (rd.random() < self.MTN_RATE):
                self.mutation(child1)
            if (rd.random() < self.MTN_RATE):
                self.mutation(child2)
            # リストに追加
            next_gene.append(child1)
            next_gene.append(child2)
        # 子供の数をオーバーしたら, 末尾を削除
        if len(next_gene) > self.CHILD_NUM:
            del next_gene[-1]
        # 世代交代 (先頭のエリート数分は残す)
        self.generation[self.ELITE_NUM:] = next_gene

    # 一定数世代を進める
    def advGeneLoop(self, loop):
        for i in range(loop):
            self.evalAndSort()  # 評価とソート
            min_dist = self.fitness[0]  # 現世代で最も短い経路距離
            print("\r最短距離: {:7.04f}".format(min_dist), end="")
            self.LOG.append(min_dist)
            self.advGene()  # 次の世代へ
        print("")

    # 適応度評価 (ただの距離計算)
    # 小さいほど良いのであまり適応度と呼びたくない
    def evalFitness(self):
        self.fitness = [self.calcPathDist(p) for p in self.generation]

    # 適応度順に並び替える
    def sortByFitness(self):
        pairs = list(zip(self.fitness, self.generation))  # (適応度, 経路) のリストを作成
        pairs.sort()  # 適応度が低い方が先頭に来るように昇順ソート
        self.generation = [p[1] for p in pairs]  # 各リストをソートしたものに置き換える
        self.fitness = [p[0] for p in pairs]

    # 評価後にソートを行う
    def evalAndSort(self):
        self.evalFitness()
        self.sortByFitness()

    # ランダムな経路を作成する関数
    def makeRandomPath(self):
        return rd.sample(range(LENGTH), LENGTH)

    # 最初の世代を作る
    def makeFirstGene(self):
        self.generation = [self.makeRandomPath()
                           for i in range(self.POPULATION)]

    # 経路の総距離を計算する
    # 個体の適応度計算で用いられると思う
    def calcPathDist(self, path):
        # 隣り合う拠点の距離の総和を計算
        return sum(self.dist_table[path[i - 1], path[i]] for i in range(LENGTH))

    # 2点間の距離を計算する
    # makeDistTable() 呼び出し用
    def calcDist(self, a, b):
        return np.sqrt(sum((self.coordinates[a] - self.coordinates[b]) ** 2))

    # 各拠点間の距離を保存する表を作る関数
    # 拠点数 × 拠点数の2次元配列
    def makeDistTable(self):
        # 全て0で初期化
        self.dist_table = np.zeros((LENGTH, LENGTH))
        for i in range(LENGTH):
            # j は必ず i より大きくする（同じ計算回避）
            for j in range(i + 1, LENGTH):
                self.dist_table[i, j] = self.calcDist(i, j)
        # 転置して足す
        # 添え字を入れ替えても同じ値になる
        self.dist_table += self.dist_table.T

    # ただ座標を確認する関数
    def viewCoordinates(self):
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        plt.show()

    # 経路を確認する関数
    def viewPath(self, path, late=True):
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        for i in range(LENGTH):  # 添え字-1が末尾を示す性質を利用して経路を描画
            ax.plot([x[path[i - 1]], x[path[i]]],
                    [y[path[i - 1]], y[path[i]]], "k-")
        if not late:  # すぐに描画する場合
            plt.show()

    # 現世代で最も短い経路を表示
    # 既にソート済みであること前提
    def viewBestPath(self, late=True):
        self.evalAndSort()
        self.viewPath(self.generation[0], late=late)

    # 経路情報を記録したファイルを作りたい
    # json形式
    def makeSampleFile(self, seed):
        np.random.seed(seed)  # シード固定
        s_arr = np.random.rand(LENGTH, 2)
        s_list = s_arr.tolist()
        print(s_list)
        s_fname = "C:\\Users\\17T2088B\\GitHub\\pracrepo\\dat\\coord100_samp01.json"
        f = open(s_fname, "w")
        json.dump(s_list, f)
        f.close()

    # 引数にサンプル数を与える(LENGTHと同じ意味)
    # the number of samples
    def makeCircle(self, nos):
        # 媒介変数 t
        t = np.arange(nos) / nos * 2 * np.pi
        x = np.cos(t)
        y = np.sin(t)
        s_fname = "C:\\Users\\17T2088B\\GitHub\\pracrepo\\dat\\circle_num{:03d}.json".format(
            nos)
        s_list = np.stack([x, y], 1).tolist()
        f = open(s_fname, "w")
        json.dump(s_list, f)
        f.close()

    # 横軸世代, 縦軸最短距離のグラフを描画
    def viewLog(self):
        x = np.arange(len(self.LOG))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, self.LOG)
        plt.show()

    # ループ数を標準入力で入力し, その数だけループさせる
    def advGeneLoopCont(self):
        loop_all = 0
        while True:
            tmp = input("ループ数: ")
            try:
                loop = int(tmp)
            except ValueError:
                loop = 0
            if loop > 0:
                self.advGeneLoop(loop)
                # self.viewBestPath()
                loop_all += loop
                print("総ループ数:", loop_all)
            else:
                break
        print("総ループ数:", loop_all)
        self.viewBestPath()
        self.viewLog()


def main():
    # ファイル読み込み
    # ランダムな点
    # fname = "C:\\Users\\17T2088B\\GitHub\\pracrepo\\dat\\coord100_samp01.json"
    # 円上の点
    #fname = "C:\\Users\\17T2088B\\GitHub\\pracrepo\\dat\\circle_num100.json"
    # f = open(fname, "r")
    # l = json.load(f)
    # f.close()
    # arr = np.array(l)
    arr = np.random.randint(0, 100, (LENGTH, 2))
    # arr = np.random.rand(LENGTH, 2)
    # print(arr)
    # 解く配列を与えてインスタンス作成
    tsp = TSP(arr)
    # tsp.crossoverTest()
    tsp.advGeneLoopCont()


if __name__ == "__main__":
    main()
