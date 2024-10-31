import numpy as np
from scipy.spatial import Delaunay
from numba import njit
import cv2
import copy

@njit
def find(x, P):
    if P[x] == x:
        return x
    P[x] = find(P[x], P)
    return P[x]

@njit
def merge2D(ux, uy, vx, vy, m, P):
    x = ux*m+uy
    y = vx*m+vy
    x = find(x, P)
    y = find(y, P)
    if x!=y:
        P[x] = y
        return True
    return False

@njit
def binToPts(bin):
    n = bin.shape[0]
    m = bin.shape[1]
    P = np.arange(n*m)
    points = []
    for x in range(n):
        for y in range(m):
            if bin[x,y]:
                if x>0 and bin[x-1,y]:
                    merge2D(x, y, x-1, y, m, P)
                if x>0 and y>0 and bin[x-1,y-1]:
                    merge2D(x, y, x-1, y-1, m, P)
                if y>0 and bin[x,y-1]:
                    merge2D(x, y, x, y-1, m, P)
                if y>0 and x<n and bin[x+1,y-1]:
                    merge2D(x, y, x+1, y-1, m, P)
                
                if x>0 and (~bin[x-1,y]) or y>0 and (~bin[x,y-1]) \
                    or x<n-1 and (~bin[x+1,y]) or y<m-1 and (~bin[x,y+1]):
                    points.append([x,y])
    points = np.array(points)
    return n, m, points, P

def imgToSts(img):
    bin = img<126
    n, m, points, P = binToPts(bin)
    
    c_p = np.zeros((n*m, 2))
    c_c = np.zeros(n*m)
    for i in range(points.shape[0]):
        x = points[i,0]
        y = points[i,1]
        j = find(x*m+y, P)
        c_p[j] += points[i]
        c_c[j] += 1

    tmp = []
    for i in range(c_c.shape[0]):
        if c_c[i]>0:
            c_p[i]=c_p[i]/c_c[i]
            tmp.append(c_p[i])
    tmp = np.array(tmp)
    
    r_l = np.zeros(n*m)
    r_c = np.zeros(n*m)
    for i in range(points.shape[0]):
        x = points[i,0]
        y = points[i,1]
        j = find(x*m+y, P)
        if x==0 or y==0 or x==n-1 or y==m-1 \
            or x>0 and (~bin[x-1,y]) or y>0 and (~bin[x,y-1]) \
            or x<n-1 and (~bin[x+1,y]) or y<m-1 and (~bin[x,y+1]):
            r_l[j] += np.sqrt(np.sum(np.square(points[i]-c_p[j])))
            r_c[j] += 1

    c_p = tmp
    tmp = []
    for i in range(r_c.shape[0]):
        if r_c[i]>0:
            r_l[i]=r_l[i]/r_c[i]
            tmp.append(r_l[i])
    r_l = np.array(tmp)
    return c_p, r_l

@njit
def convert(img):
    # 并查集
    bin = img<126
    n, m, points, P = binToPts(bin)
    return points.astype(np.float64), P

@njit
def calMinAlpha(tri, points):
    e = []
    l = []
    for t in tri:
        p0 = t[0]
        p1 = t[1]
        p2 = t[2]
        e.append((p0,p1))
        e.append((p0,p2))
        e.append((p1,p2))
        p0 = points[p0]
        p1 = points[p1]
        p2 = points[p2]
        l1 = np.sqrt(np.sum(np.square(p0-p1)))
        l2 = np.sqrt(np.sum(np.square(p0-p2)))
        l3 = np.sqrt(np.sum(np.square(p1-p2)))
        l.append(l1)
        l.append(l2)
        l.append(l3)
    l = np.array(l)
    e = np.array(e)
    idx = np.argsort(l)
    p = np.arange(points.shape[0])
    a_t = []
    for i in range(e.shape[0]):
        j = idx[i]
        u = find(e[j][0], p)
        v = find(e[j][1], p)
        if u!=v:
            p[u] = v
            a_t.append(l[j])
    a_t = a_t[-max(int(len(a_t)*0.04), 1)]
    return a_t

def geometry(points: np.ndarray, tri: np.ndarray):
    t0 = tri[:,0]
    t1 = tri[:,1]
    t2 = tri[:,2]
    p0 = points[t0]
    p1 = points[t1]
    p2 = points[t2]

    a = p0-p1
    b = p2-p1
    c = p2-p0

    a_squ = np.sum(np.square(a),1)
    b_squ = np.sum(np.square(b),1)
    c_squ = np.sum(np.square(c),1)
    a_len = np.sqrt(a_squ)
    b_len = np.sqrt(b_squ)

    t_a = np.zeros((a_squ.shape[0],))
    cosc = (a_squ+b_squ-c_squ)/(2*a_len*b_len)
    idx = (cosc<1)&(cosc>-1)
    tri = tri[idx]
    #三角形顶点目录
    t0 = tri[:,0]
    t1 = tri[:,1]
    t2 = tri[:,2]
    p0 = points[t0]
    p1 = points[t1]
    p2 = points[t2]

    a = p0-p1
    b = p2-p1
    c = p2-p0

    a_squ = np.sum(np.square(a),1)
    b_squ = np.sum(np.square(b),1)
    c_squ = np.sum(np.square(c),1)
    a_len = np.sqrt(a_squ)
    b_len = np.sqrt(b_squ)
    c_len = np.sqrt(c_squ)

    d=(a_len+b_len+c_len)/2
    area=np.sqrt(d*(d-a_len)*(d-b_len)*(d-c_len))

    t_a = np.zeros((a_squ.shape[0],))
    cosc = (a_squ+b_squ-c_squ)/(2*a_len*b_len)
    sinc = np.sqrt(1-cosc*cosc)

    d = np.sqrt(c_squ)/sinc
    t_a = d/2
    #使每条边都有归属
    idx = np.argsort(t_a)
    tri = tri[idx]
    t_a = t_a[idx]
    area = area[idx]
    return t_a, tri, area

class Extractor:
    def __init__(self, N):
        self.cnt = np.zeros(N)
        self.G = []
        self.vis = np.zeros(N)
        self.sum = 0
        for i in range(N):
            self.G.append([])
    
    def dealEdge(self, u: np.int32,v: np.int32):
        G = self.G
        cnt = self.cnt
        if v not in G[u]:
            G[u].append(v)
            G[v].append(u)
            cnt[u] += 1
            cnt[v] += 1
        else:
            G[u].remove(v)
            G[v].remove(u)
            cnt[u] -= 1
            cnt[v] -= 1
    
    def visit(self, p):
        if self.vis[p]==0:
            self.sum += 1
        self.vis[p] += 1

    def leave(self, p):
        self.vis[p] -= 1
        if self.vis[p] == 0:
            self.sum -= 1

    def addTri(self, t):
        p0 = t[0]
        p1 = t[1]
        p2 = t[2]
        self.dealEdge(p0, p1)
        self.dealEdge(p0, p2)
        self.dealEdge(p1, p2)
        self.visit(p0)
        self.visit(p1)
        self.visit(p2)

    def delTri(self, t):
        p0 = t[0]
        p1 = t[1]
        p2 = t[2]
        self.dealEdge(p0, p1)
        self.dealEdge(p0, p2)
        self.dealEdge(p1, p2)
        self.leave(p0)
        self.leave(p1)
        self.leave(p2)

    def polygon(self, points):
        G = copy.deepcopy(self.G)
        poly = []
        for i in range(len(G)):
            if len(G[i]) > 0:
                # o是当前轮廓
                p = []
                u = i
                v = G[u][-1]
                p.append(points[u])
                p.append(points[v])
                del G[u][-1]
                while len(G[v]) > 0:
                    G[v].remove(u)
                    if len(G[v]) > 0:
                        u = v
                        v = G[u][-1]
                        p.append(points[v])
                        del G[u][-1]
                p = np.array(p)
                poly.append(p)
        return poly
    
def calAlpha(points: np.ndarray, lbd):
    from scipy.spatial import Delaunay
    
    tri = Delaunay(points).simplices
    t_a, tri, area = geometry(points, tri)
    for i in range(tri.shape[0]-1):
        area[i+1] += area[i]

    ext = Extractor(points.shape[0])
    a_x = []
    a_y = []
    c = 0
    a = 0
    g_r = None
    res = None
    g_min = 1e8
    for i in range(tri.shape[0]):
        ext.addTri(tri[i])
        if i==tri.shape[0]-1 or t_a[i+1]>t_a[i]+1e-6:
            if c>0:
                g_r = np.sqrt(a/np.pi*(1-np.power(lbd, 1/c)))-t_a[i]
                if np.abs(g_r)<g_min:
                    g_min = np.abs(g_r)
                    res = t_a[i]
                a_x.append(t_a[i])
                a_y.append(g_r)
                if g_l*g_r<0:
                    g_min = 0
                    res = -(a_x[-1]-a_x[-2])/(g_r-g_l)*g_l+a_x[-2]
            c = ext.sum

            a = area[i]
            g_l = np.sqrt(a/np.pi*(1-np.power(lbd, 1/c)))-t_a[i]
            if np.abs(g_l)<g_min:
                g_min = np.abs(g_l)
                res = t_a[i]+1e-6
            a_x.append(t_a[i])
            a_y.append(g_l)
            if g_r is not None and g_l*g_r<=0:
                g_min = 0
                res = t_a[i]
    if res is None:
        res = calMinAlpha(tri, points)
    else:
        res = max(calMinAlpha(tri, points), res)
    return res

def paintPly(polys, h0, w0, h:int = None, w:int = None, lw: int = 1, lc = 0, bc = 255):
    """
    :param h0,w0: 原图长宽
    :param h,w: 目标图长宽
    :param bc: 背景颜色
    """

    if (h is None) and (w is None):
        h = h0
        w = w0
    elif h is None:
        h = int(h0*w/w0+0.5)
    elif w is None:
        w = int(w0*h/h0+0.5)
    truth =  np.full((w, h, 3), bc, dtype = np.uint8)
    for j in range(len(polys)):
        points = polys[j]
        if h!=h0:
            points[:,0] = points[:,0]*h/h0
        if w!=w0:
            points[:,1] = points[:,1]*w/w0
        polys[j] = points
        
    img = _paintPly(truth, polys, lc)
    img = img.swapaxes(0,1)
    if lw>1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (lw, lw))
        img = cv2.erode(img, kernel)
    return img

def _paintPly(img, polygon: list, lc):
    for poly in polygon:
        py = poly.astype(np.int32)
        l = 0
        line = []
        for i in range(len(py)-1):
            line.append(np.sqrt(np.sum(np.square(py[i]-py[i+1]))))
            l += line[-1]
        y = []
        itv = l/400
        s = 0
        for i in range(len(py)-1):
            e = line[i]
            while s<e:
                p = py[i]+s/e*(py[i+1]-py[i])
                y.append(p)
                s += itv
            s -= e
        y.append(py[-1])
        y = np.stack(y,0)
       
        for i in range(5):
            v = np.insert(y, 0, y[399:400], axis=0)
            v = np.append(v, y[1:2], axis=0)
            v = v[1:]-v[:-1]
            l = np.sqrt(np.sum(np.square(v),1))
            t = (v[:-1,0]*v[1:,0]+v[:-1,1]*v[1:,1])/(l[:-1]*l[1:])
            t = np.minimum(t,1)
            a = np.arccos(t)
            for j in range(401):
                y[j] = y[j]+np.exp(-np.abs(a[j]-np.pi))*(y[(j+402)%401]-2*y[j]+y[(j+400)%401])

        for i in range(400):
            pt1 = (int(y[i,0]), int(y[i,1]))
            pt2 = (int(y[(i+1)%400,0]), int(y[(i+1)%400,1]))
            cv2.line(img, pt1, pt2, (lc, lc, lc))
    return img

def edge(img, lbd):
    points, radius = imgToSts(img)
    alpha = calAlpha(points, lbd)
    radius = np.mean(radius)
    alpha -= radius
    alpha = max(alpha, 0)
    
    # print('alpha', alpha, radius)
    points, P = convert(img)
    tri = Delaunay(points).simplices
    t_a, tri, area = geometry(points, tri)
    ext = Extractor(points.shape[0])
    vis = np.zeros((img.shape[0]*img.shape[1]))
    R = np.array(P)
    for i in range(tri.shape[0]):
        if alpha>t_a[i]:
            x = points[tri[i,0]].astype(np.int32)
            y = points[tri[i,1]].astype(np.int32)
            z = points[tri[i,2]].astype(np.int32)
            x = find(x[0]*img.shape[1]+x[1],R)
            y = find(y[0]*img.shape[1]+y[1],R)
            if x!=y:
                R[x] = y
                vis[y] = 1
            z = find(z[0]*img.shape[1]+z[1],R)
            if y!=z:
                R[y] = z
                vis[z] = 1
    # print(np.sum(vis))
    for i in range(tri.shape[0]):
        x = points[tri[i,0]].astype(np.int32)
        y = points[tri[i,1]].astype(np.int32)
        z = points[tri[i,2]].astype(np.int32)
        x = find(x[0]*img.shape[1]+x[1],P)
        y = find(y[0]*img.shape[1]+y[1],P)
        z = find(z[0]*img.shape[1]+z[1],P)
        if alpha>t_a[i] or (x==y and y==z):
            x = points[tri[i,0]].astype(np.int32)
            x = find(x[0]*img.shape[1]+x[1],R)
            if vis[x]>0:
                ext.addTri(tri[i])
    poly = ext.polygon(points)
    img = paintPly(poly, img.shape[0], img.shape[1])
    return img