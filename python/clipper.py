#===============================================================================
#                                                                              #
# Author    :  Angus Johnson                                                   #
# Version   :  5.1.6                                                           #
# Date      :  23 May 2013                                                     #
# Website   :  http://www.angusj.com                                           #
# Copyright :  Angus Johnson 2010-2013                                         #
#                                                                              #
# License:                                                                     #
# Use, modification & distribution is subject to Boost Software License Ver 1. #
# http://www.boost.org/LICENSE_1_0.txt                                         #
#                                                                              #
# Attributions:                                                                #
# The code in this library is an extension of Bala Vatti's clipping algorithm: #
# "A generic solution to polygon clipping"                                     #
# Communications of the ACM, Vol 35, Issue 7 (July 1992) PP 56-63.             #
# http://portal.acm.org/citation.cfm?id=129906                                 #
#                                                                              #
# Computer graphics and geometric modeling: implementation and algorithms      #
# By Max K. Agoston                                                            #
# Springer; 1 edition (January 4, 2005)                                        #
# http://books.google.com/books?q=vatti+clipping+agoston                       #
#                                                                              #
# See also:                                                                    #
# "Polygon Offsetting by Computing Winding Numbers"                            #
# Paper no. DETC2005-85513 PP. 565-575                                         #
# ASME 2005 International Design Engineering Technical Conferences             #
# and Computers and Information in Engineering Conference (IDETC/CIE2005)      #
# September 24-28, 2005 , Long Beach, California, USA                          #
# http://www.me.berkeley.edu/~mcmains/pubs/DAC05OffsetPolygon.pdf              #
#                                                                              #
#===============================================================================

import math
from collections import namedtuple
from decimal import Decimal, getcontext

getcontext().prec = 8
horizontal = Decimal('-Infinity')

class ClipType: (Intersection, Union, Difference, Xor) = range(4)
class PolyType:    (Subject, Clip) = range(2)
class PolyFillType: (EvenOdd, NonZero, Positive, Negative) = range(4)
class JoinType: (Square, Round, Miter) = range(3)
class EndType: (Closed, Butt, Square, Round) = range(4)
class EdgeSide: (Left, Right) = range(2)
class Protects: (Neither, Left, Right, Both) = range(4)
class Direction: (LeftToRight, RightToLeft) = range(2)

Point = namedtuple('Point', 'x y')
DoublePoint = namedtuple('DoublePoint', 'x y')

class LocalMinima(object):
    leftBound = rightBound = nextLm = None
    def __init__(self, y, leftBound, rightBound):
        self.y = y
        self.leftBound = leftBound
        self.rightBound = rightBound

class Scanbeam(object):
    __slots__ = ('y','nextSb')
    def __init__(self, y, nextSb = None):
        self.y = y
        self.nextSb = nextSb
    def __repr__(self):
        s = 'None'
        if self.nextSb is not None: s = '<obj>'
        return "(y:%i, nextSb:%s)" % (self.y, s)

class IntersectNode(object):
    __slots__ = ('e1','e2','pt','nextIn')
    def __init__(self, e1, e2, pt):
        self.e1 = e1
        self.e2 = e2
        self.pt = pt
        self.nextIn = None

class OutPt(object):
    __slots__ = ('idx','pt','prevOp','nextOp')
    def __init__(self, idx, pt):
        self.idx = idx
        self.pt = pt
        self.prevOp = None
        self.nextOp = None

class OutRec(object):
    __slots__ = ('idx','bottomPt','isHole','FirstLeft', 'pts','PolyNode')
    def __init__(self, idx):
        self.idx = idx
        self.bottomPt = None
        self.isHole = False
        self.FirstLeft = None
        self.pts = None
        self.PolyNode = None

class JoinRec(object):
    __slots__ = ('pt1a','pt1b','poly1Idx','pt2a', 'pt2b','poly2Idx')

class HorzJoin(object):
    edge = None
    savedIdx = 0
    prevHj = None
    nextHj = None
    def __init__(self, edge, idx):
        self.edge = edge
        self.savedIdx = idx

#===============================================================================
# Unit global functions ...
#===============================================================================
def IntsToPoints(ints):
    result = []
    for i in range(0, len(ints), 2):
        result.append(Point(ints[i], ints[i+1]))
    return result

def Area(polygon):
    # see http://www.mathopenref.com/coordpolygonarea2.html
    highI = len(polygon) - 1
    A = (polygon[highI].x + polygon[0].x) * (polygon[0].y - polygon[highI].y)
    for i in range(highI):
        A += (polygon[i].x + polygon[i+1].x) * (polygon[i+1].y - polygon[i].y)
    return float(A) / 2

def Orientation(polygon):
    return Area(polygon) > 0.0

#===============================================================================
# PolyNode & PolyTree classes (+ ancilliary functions)
#===============================================================================
class PolyNode(object):
    """Node of PolyTree"""
    
    def __init__(self):
        self.Contour = []
        self.Childs = []
        self.Parent = None
        self.Index = 0
        self.ChildCount = 0
    
    def IsHole(self):
        result = True
        while (self.Parent is not None):
            result = not result
            self.Parent = self.Parent.Parent
        return result
    
    def GetNext(self):
        if (self.ChildCount > 0):
            return self.Childs[0]
        else:
            return self._GetNextSiblingUp()
    
    def _AddChild(self, node):
        self.Childs.append(node)
        node.Index = self.ChildCount
        node.Parent = self
        self.ChildCount += 1
    
    def _GetNextSiblingUp(self):
        if (self.Parent is None):
            return None
        elif (self.Index == self.Parent.ChildCount - 1):
            return self.Parent._GetNextSiblingUp()
        else:
            return self.Parent.Childs[self.Index +1]

class PolyTree(PolyNode):
    """Container for PolyNodes"""

    def __init__(self):
        PolyNode.__init__(self)
        self._AllNodes = []
        
    def Clear(self):
        self._AllNodes = []
        self.Childs = []
        self.ChildCount = 0
    
    def GetFirst(self):
        if (self.ChildCount > 0):
            return self.Childs[0]
        else:
            return None
    
    def Total(self):
        return len(self._AllNodes)

def _AddPolyNodeToPolygons(polynode, polygons):
    """Internal function for PolyTreeToPolygons()"""
    if (len(polynode.Contour) > 0):
        polygons.append(polynode.Contour)
    for i in range(polynode.ChildCount):
        _AddPolyNodeToPolygons(polynode.Childs[i], polygons)

def PolyTreeToPolygons(polyTree):
    result = []
    _AddPolyNodeToPolygons(polyTree, result)
    return result

#===============================================================================
# Edge class 
#===============================================================================

class Edge(object):

    def __init__(self):
        self.xBot, self.yBot, self.xCurr, self.yCurr, = 0, 0, 0, 0
        self.xTop, self.yTop = 0, 0
        self.dx, self.deltaX , self.deltaY = Decimal(0), Decimal(0), Decimal(0)
        self.polyType = PolyType.Subject 
        self.side = EdgeSide.Left
        self.windDelta, self.windCnt, self.windCnt2 = 0, 0, 0 
        self.outIdx = -1
        self.nextE, self.prevE, self.nextInLML = None, None, None
        self.prevInAEL, self.nextInAEL, self.prevInSEL, self.nextInSEL = None, None, None, None
        
    def __repr__(self):
        return "(%i,%i . %i,%i {dx:%0.2f} %i {%x})" % \
            (self.xBot, self.yBot, self.xTop, self.yTop, self.dx, self.outIdx, id(self))

#===============================================================================
# ClipperBase class (+ data structs & ancilliary functions)
#===============================================================================

def _PointsEqual(pt1, pt2):
    return (pt1.x == pt2.x) and (pt1.y == pt2.y)

def _SlopesEqual(pt1, pt2, pt3, pt4 = None):
    if pt4 is None:
        return (pt1.y-pt2.y)*(pt2.x-pt3.x) == (pt1.x-pt2.x)*(pt2.y-pt3.y)
    else:
        return (pt1.y-pt2.y)*(pt3.x-pt4.x) == (pt1.x-pt2.x)*(pt3.y-pt4.y)

def _SlopesEqual2(e1, e2):
    return e1.deltaY * e2.deltaX == e1.deltaX * e2.deltaY

def _SetDx(e):
    e.deltaX = Decimal(e.xTop - e.xBot)
    e.deltaY = Decimal(e.yTop - e.yBot)
    if e.deltaY == 0: e.dx = horizontal
    else: e.dx = e.deltaX/e.deltaY

def _SwapSides(e1, e2):
    side    = e1.side
    e1.side = e2.side
    e2.side = side

def _SwapPolyIndexes(e1, e2):
    idx       = e1.outIdx
    e1.outIdx = e2.outIdx
    e2.outIdx = idx

def _InitEdge(e, eNext, ePrev, pt, polyType):
    e.nextE = eNext
    e.prevE = ePrev
    e.xCurr = pt.x
    e.yCurr = pt.y
    if e.yCurr >= e.nextE.yCurr:
        e.xBot = e.xCurr
        e.yBot = e.yCurr
        e.xTop = e.nextE.xCurr
        e.yTop = e.nextE.yCurr
        e.windDelta = 1
    else:
        e.xTop = e.xCurr
        e.yTop = e.yCurr
        e.xBot = e.nextE.xCurr
        e.yBot = e.nextE.yCurr
        e.windDelta = -1
    _SetDx(e)
    e.outIdx = -1
    e.PolyType = polyType

def _SwapX(e):
    e.xCurr = e.xTop
    e.xTop = e.xBot
    e.xBot = e.xCurr
    
class ClipperBase(object):

    def __init__(self):
        self._EdgeList      = []       # 2D array
        self._LocalMinList  = None     # single-linked list of LocalMinima
        self._CurrentLocMin = None
        
    def _InsertLocalMinima(self, lm):
        if self._LocalMinList is None:
            self._LocalMinList = lm
        elif lm.y >= self._LocalMinList.y:
            lm.nextLm = self._LocalMinList
            self._LocalMinList = lm
        else:
            tmp = self._LocalMinList
            while tmp.nextLm is not None and lm.y < tmp.nextLm.y:
                    tmp = tmp.nextLm
            lm.nextLm = tmp.nextLm
            tmp.nextLm = lm

    def _AddBoundsToLML(self, e):
        e.nextInLML = None
        e = e.nextE
        while True:
            if e.dx == horizontal:
                if (e.nextE.yTop < e.yTop) and (e.nextE.xBot > e.prevE.xBot): break
                if (e.xTop != e.prevE.xBot): _SwapX(e)
                e.nextInLML = e.prevE
            elif e.yBot == e.prevE.yBot: break
            else: e.nextInLML = e.prevE
            e = e.nextE

        if e.dx == horizontal:
            if (e.xBot != e.prevE.xBot): _SwapX(e)
            lm = LocalMinima(e.prevE.yBot, e.prevE, e)
        elif (e.dx < e.prevE.dx):
            lm = LocalMinima(e.prevE.yBot, e.prevE, e)
        else:
            lm = LocalMinima(e.prevE.yBot, e, e.prevE)
        lm.leftBound.side = EdgeSide.Left
        lm.rightBound.side = EdgeSide.Right
        self._InsertLocalMinima(lm)
        while True:
            if e.nextE.yTop == e.yTop and e.nextE.dx != horizontal: break
            e.nextInLML = e.nextE
            e = e.nextE
            if e.dx == horizontal and e.xBot != e.prevE.xTop: _SwapX(e)
        return e.nextE

    def _Reset(self):
        lm = self._LocalMinList
        if lm is not None: self._CurrentLocMin = lm
        while lm is not None:
            e = lm.leftBound
            while e is not None:
                e.xCurr    = e.xBot
                e.yCurr    = e.yBot
                e.side     = EdgeSide.Left
                e.outIdx = -1
                e = e.nextInLML
            e = lm.rightBound
            while e is not None:
                e.xCurr    = e.xBot
                e.yCurr    = e.yBot
                e.side     = EdgeSide.Right
                e.outIdx = -1
                e = e.nextInLML
            lm = lm.nextLm
            
    def AddPolygon(self, polygon, polyType):
        ln = len(polygon)
        if ln < 3: return False
        pg = polygon[:]
        j = 0
        # remove duplicate points and co-linear points
        for i in range(1, len(polygon)):
            if _PointsEqual(pg[j], polygon[i]): 
                continue
            elif (j > 0) and _SlopesEqual(pg[j-1], pg[j], polygon[i]):
                if _PointsEqual(pg[j-1], polygon[i]): j -= 1
            else: j += 1
            pg[j] = polygon[i]
        if (j < 2): return False
        # remove duplicate points and co-linear edges at the loop around
        # of the start and end coordinates ...
        ln = j +1
        while (ln > 2):
            if _PointsEqual(pg[j], pg[0]): j -= 1
            elif _PointsEqual(pg[0], pg[1]) or _SlopesEqual(pg[j], pg[0], pg[1]):
                pg[0] = pg[j]
                j -= 1
            elif _SlopesEqual(pg[j-1], pg[j], pg[0]): j -= 1
            elif _SlopesEqual(pg[0], pg[1], pg[2]):
                for i in range(2, j +1): pg[i-1] = pg[i]
                j -= 1
            else: break
            ln -= 1
        if ln < 3: return False
        edges = []
        for i in range(ln):
            edges.append(Edge())
        edges[0].xCurr = pg[0].x
        edges[0].yCurr = pg[0].y
        _InitEdge(edges[ln-1], edges[0], edges[ln-2], pg[ln-1], polyType)
        for i in range(ln-2, 0, -1):
            _InitEdge(edges[i], edges[i+1], edges[i-1], pg[i], polyType)
        _InitEdge(edges[0], edges[1], edges[ln-1], pg[0], polyType)
        e = edges[0]
        eHighest = e
        while True:
            e.xCurr = e.xBot
            e.yCurr = e.yBot
            if e.yTop < eHighest.yTop: eHighest = e
            e = e.nextE
            if e == edges[0]: break
        # make sure eHighest is positioned so the following loop works safely ...
        if eHighest.windDelta > 0: eHighest = eHighest.nextE
        if eHighest.dx == horizontal: eHighest = eHighest.nextE
        # finally insert each local minima ...
        e = eHighest
        while True:
            e = self._AddBoundsToLML(e)
            if e == eHighest: break
        self._EdgeList.append(edges)

    def AddPolygons(self, polygons, polyType):
        result = False
        for p in polygons:
            if self.AddPolygon(p, polyType): result = True
        return result

    def Clear(self):
        self._EdgeList = []
        self._LocalMinList    = None
        self._CurrentLocMin = None

    def _PopLocalMinima(self):
        if self._CurrentLocMin is not None:
            self._CurrentLocMin = self._CurrentLocMin.nextLm

#===============================================================================
# Clipper class (+ data structs & ancilliary functions)
#===============================================================================
def _IntersectPoint(edge1, edge2):
    if _SlopesEqual2(edge1, edge2):
        if (edge2.ybot > edge1.ybot): y = edge2.ybot 
        else: y = edge1.ybot
        return Point(0, y), False
    if edge1.dx == 0:
        x = edge1.xBot
        if edge2.dx == horizontal:
            y = edge2.yBot
        else:
            b2 = edge2.yBot - Decimal(edge2.xBot)/edge2.dx
            y = round(Decimal(x)/edge2.dx + b2)
    elif edge2.dx == 0:
        x = edge2.xBot
        if edge1.dx == horizontal:
            y = edge1.yBot
        else:
            b1 = edge1.yBot - Decimal(edge1.xBot)/edge1.dx
            y = round(Decimal(x)/edge1.dx + b1)
    else:
        b1 = edge1.xBot - edge1.yBot * edge1.dx
        b2 = edge2.xBot - edge2.yBot * edge2.dx
        m    = Decimal(b2-b1)/(edge1.dx - edge2.dx)
        y    = round(m)
        if math.fabs(edge1.dx) < math.fabs(edge2.dx):
            x = round(edge1.dx * m + b1)
        else:
            x = round(edge2.dx * m + b2)
    if (y < edge1.yTop) or (y < edge2.yTop):
        if (edge1.yTop > edge2.yTop):
            return Point(edge1.xTop,edge1.yTop), _TopX(edge2, edge1.yTop) < edge1.xTop
        else:
            return Point(edge2.xTop,edge2.yTop), _TopX(edge1, edge2.yTop) > edge2.xTop
    else:
        return Point(x,y), True

def _TopX(e, currentY):
    if currentY == e.yTop: return e.xTop
    elif e.xTop == e.xBot: return e.xBot
    else: return e.xBot + round(e.dx * Decimal(currentY - e.yBot))

def _E2InsertsBeforeE1(e1,e2):
    if (e2.xCurr == e1.xCurr): 
        if (e2.yTop > e1.yTop):
            return e2.xTop < _TopX(e1, e2.yTop) 
        return e1.xTop > _TopX(e2, e1.yTop) 
    else: 
        return e2.xCurr < e1.xCurr

def _IsMinima(e):
    return e is not None and e.prevE.nextInLML != e and e.nextE.nextInLML != e

def _IsMaxima(e, y):
    return e is not None and e.yTop == y and e.nextInLML is None

def _IsIntermediate(e, y):
    return e.yTop == y and e.nextInLML is not None

def _GetMaximaPair(e):
    if not _IsMaxima(e.nextE, e.yTop) or e.nextE.xTop != e.xTop:
        return e.prevE
    else:
        return e.nextE

def _GetnextInAEL(e, direction):
    if direction == Direction.LeftToRight: return e.nextInAEL
    else: return e.prevInAEL

def _ProtectLeft(val):
    if val: return Protects.Both
    else: return Protects.Right

def _ProtectRight(val):
    if val: return Protects.Both
    else: return Protects.Left

def _GetDx(pt1, pt2):
    if (pt1.y == pt2.y): return horizontal
    else: return Decimal(pt2.x - pt1.x)/(pt2.y - pt1.y)

def _Param1RightOfParam2(outRec1, outRec2):
    while outRec1 is not None:
        outRec1 = outRec1.FirstLeft
        if outRec1 == outRec2: return True
    return False

def _FirstParamIsbottomPt(btmPt1, btmPt2):
    p = btmPt1.prevOp
    while _PointsEqual(p.pt, btmPt1.pt) and (p != btmPt1): p = p.prevOp
    dx1p = abs(_GetDx(btmPt1.pt, p.pt))
    p = btmPt1.nextOp
    while _PointsEqual(p.pt, btmPt1.pt) and (p != btmPt1): p = p.nextOp
    dx1n = abs(_GetDx(btmPt1.pt, p.pt))

    p = btmPt2.prevOp
    while _PointsEqual(p.pt, btmPt2.pt) and (p != btmPt2): p = p.prevOp
    dx2p = abs(_GetDx(btmPt2.pt, p.pt))
    p = btmPt2.nextOp
    while _PointsEqual(p.pt, btmPt2.pt) and (p != btmPt2): p = p.nextOp
    dx2n = abs(_GetDx(btmPt2.pt, p.pt))
    return (dx1p >= dx2p and dx1p >= dx2n) or (dx1n >= dx2p and dx1n >= dx2n)

def _GetBottomPt(pp):
    dups = None
    p = pp.nextOp
    while p != pp:
        if p.pt.y > pp.pt.y:
            pp = p
            dups = None
        elif p.pt.y == pp.pt.y and p.pt.x <= pp.pt.x:
            if p.pt.x < pp.pt.x:
                dups = None
                pp = p
            else:
                if p.nextOp != pp and p.prevOp != pp: dups = p
        p = p.nextOp
    if dups is not None:
        while dups != p:
            if not _FirstParamIsbottomPt(p, dups): pp = dups
            dups = dups.nextOp
            while not _PointsEqual(dups.pt, pp.pt): dups = dups.nextOp
    return pp

def _GetLowermostRec(outRec1, outRec2):
    if (outRec1.bottomPt is None): 
        outPt1 = _GetBottomPt(outRec1.pts)
    else: outPt1 = outRec1.bottomPt
    if (outRec2.bottomPt is None): 
        outPt2 = _GetBottomPt(outRec2.pts)
    else: outPt2 = outRec2.bottomPt
    if (outPt1.pt.y > outPt2.pt.y): return outRec1
    elif (outPt1.pt.y < outPt2.pt.y): return outRec2
    elif (outPt1.pt.x < outPt2.pt.x): return outRec1
    elif (outPt1.pt.x > outPt2.pt.x): return outRec2
    elif (outPt1.nextOp == outPt1): return outRec2
    elif (outPt2.nextOp == outPt2): return outRec1
    elif _FirstParamIsbottomPt(outPt1, outPt2): return outRec1
    else: return outRec2

def _SetHoleState(e, outRec, polyOutList):
    isHole = False
    e2 = e.prevInAEL
    while e2 is not None:
        if e2.outIdx >= 0:
            isHole = not isHole
            if outRec.FirstLeft is None:
                outRec.FirstLeft = polyOutList[e2.outIdx]
        e2 = e2.prevInAEL
    outRec.isHole = isHole

def _PointCount(pts):
    if pts is None: return 0
    p = pts
    result = 0
    while True:
        result += 1
        p = p.nextOp
        if p == pts: break
    return result

def _PointIsVertex(pt, outPts):
    op = outPts
    while True:
        if _PointsEqual(op.pt, pt): return True
        op = op.nextOp
        if op == outPts: break
    return False
               
def _ReversePolyPtLinks(pp):
    if pp is None: return
    pp1 = pp
    while True:
        pp2 = pp1.nextOp
        pp1.nextOp = pp1.prevOp
        pp1.prevOp = pp2;
        pp1 = pp2
        if pp1 == pp: break

def _FixupOutPolygon(outRec):
    lastOK = None
    outRec.bottomPt = None
    pp = outRec.pts
    while True:
        if pp.prevOp == pp or pp.nextOp == pp.prevOp:
            outRec.pts = None
            return
        if _PointsEqual(pp.pt, pp.nextOp.pt) or \
                _SlopesEqual(pp.prevOp.pt, pp.pt, pp.nextOp.pt):
            lastOK = None
            pp.prevOp.nextOp = pp.nextOp
            pp.nextOp.prevOp = pp.prevOp
            pp = pp.prevOp
        elif pp == lastOK: break
        else:
            if lastOK is None: lastOK = pp
            pp = pp.nextOp
    outRec.pts = pp

def _FixHoleLinkage(outRec):
    if outRec.FirstLeft is None or \
        (outRec.isHole != outRec.FirstLeft.isHole and \
            outRec.FirstLeft.pts is not None): return
    orfl = outRec.FirstLeft
    while orfl is not None and \
            (orfl.isHole == outRec.isHole or orfl.pts is None):
        orfl = orfl.FirstLeft
    outRec.FirstLeft = orfl
    
def _GetOverlapSegment(pt1a, pt1b, pt2a, pt2b):
    # precondition: segments are co-linear
    if abs(pt1a.x - pt1b.x) > abs(pt1a.y - pt1b.y):
        if pt1a.x > pt1b.x: tmp = pt1a; pt1a = pt1b; pt1b = tmp
        if pt2a.x > pt2b.x: tmp = pt2a; pt2a = pt2b; pt2b = tmp
        if (pt1a.x > pt2a.x): pt1 = pt1a
        else: pt1 = pt2a
        if (pt1b.x < pt2b.x): pt2 = pt1b
        else: pt2 = pt2b
        return pt1, pt2, pt1.x < pt2.x
    else:
        if pt1a.y < pt1b.y: tmp = pt1a; pt1a = pt1b; pt1b = tmp 
        if pt2a.y < pt2b.y: tmp = pt2a; pt2a = pt2b; pt2b = tmp
        if (pt1a.y < pt2a.y): pt1 = pt1a 
        else: pt1 = pt2a
        if (pt1b.y > pt2b.y): pt2 = pt1b 
        else: pt2 = pt2b
        return pt1, pt2, pt1.y > pt2.y

    
def _FindSegment(outPt, pt1, pt2):
    if outPt is None: return outPt, pt1, pt2, False
    pt1a = pt1; pt2a = pt2
    outPt2 = outPt
    while True:
        if _SlopesEqual(pt1a, pt2a, outPt.pt, outPt.prevOp.pt) and _SlopesEqual(pt1a, pt2a, outPt.pt):
            pt1, pt2, overlap = _GetOverlapSegment(pt1a, pt2a, outPt.pt, outPt.prevOp.pt)
            if overlap: return outPt, pt1, pt2, True
        outPt = outPt.nextOp
        if outPt == outPt2: return outPt, pt1, pt2, False

def _Pt3IsBetweenPt1AndPt2(pt1, pt2, pt3):
    if _PointsEqual(pt1, pt3) or _PointsEqual(pt2, pt3): return True
    elif pt1.x != pt2.x: return (pt1.x < pt3.x) == (pt3.x < pt2.x)
    else: return (pt1.y < pt3.y) == (pt3.y < pt2.y)

def _InsertPolyPtBetween(outPt1, outPt2, pt):
    if outPt1 == outPt2: raise Exception("JoinError")
    result = OutPt(outPt1.idx, pt)
    if outPt2 == outPt1.nextOp:
        outPt1.nextOp = result
        outPt2.prevOp = result
        result.nextOp = outPt2
        result.prevOp = outPt1
    else:
        outPt2.nextOp = result
        outPt1.prevOp = result
        result.nextOp = outPt1
        result.prevOp = outPt2
    return result

def _PointOnLineSegment(pt, linePt1, linePt2):
    return ((pt.x == linePt1.x) and (pt.y == linePt1.y)) or \
        ((pt.x == linePt2.x) and (pt.y == linePt2.y)) or \
        (((pt.x > linePt1.x) == (pt.x < linePt2.x)) and \
        ((pt.y > linePt1.y) == (pt.y < linePt2.y)) and \
        ((pt.x - linePt1.x) * (linePt2.y - linePt1.y) == \
        (linePt2.x - linePt1.x) * (pt.y - linePt1.y)))

def _PointOnPolygon(pt, pp):
    pp2 = pp;
    while True:
        if (_PointOnLineSegment(pt, pp2.pt, pp2.nextOp.pt)):
            return True
        pp2 = pp2.nextOp
        if (pp2 == pp): return False

def _PointInPolygon(pt, outPt): 
    result = False
    outPt2 = outPt
    while True:
        if ((((outPt2.pt.y <= pt.y) and (pt.y < outPt2.prevOp.pt.y)) or \
            ((outPt2.prevOp.pt.y <= pt.y) and (pt.y < outPt2.pt.y))) and \
            (pt.x < (outPt2.prevOp.pt.x - outPt2.pt.x) * (pt.y - outPt2.pt.y) / \
            (outPt2.prevOp.pt.y - outPt2.pt.y) + outPt2.pt.x)): result = not result
        outPt2 = outPt2.nextOp
        if outPt2 == outPt: break

def _Poly2ContainsPoly1(outPt1, outPt2):
    pt = outPt1
    if (_PointOnPolygon(pt.pt, outPt2)):
        pt = pt.nextOp
        while (pt != outPt1 and _PointOnPolygon(pt.pt, outPt2)):
            pt = pt.nextOp
        if (pt == outPt1): return True
    return _PointInPolygon(pt.pt, outPt2)    
    
def _EdgesAdjacent(inode):
    return (inode.e1.nextInSEL == inode.e2) or \
        (inode.e1.prevInSEL == inode.e2)

def _UpdateOutPtIdxs(outrec):
    op = outrec.pts
    while True:
        op.idx = outrec.idx
        op = op.prevOp
        if (op == outrec.pts): break

class Clipper(ClipperBase):

    def __init__(self):
        ClipperBase.__init__(self)

        self.ReverseOutput     = False
        self.ForceSimple       = False
        
        self._PolyOutList = []        
        self._ClipType         = ClipType.Intersection
        self._Scanbeam         = None
        self._ActiveEdges      = None
        self._SortedEdges      = None
        self._IntersectNodes   = None
        self._ClipFillType     = PolyFillType.EvenOdd
        self._SubjFillType     = PolyFillType.EvenOdd
        self._ExecuteLocked    = False
        self._UsingPolyTree    = False
        self._JoinList         = None
        self._HorzJoins        = None
        
    def _Reset(self):
        ClipperBase._Reset(self)
        self._Scanbeam = None
        self._PolyOutList = []
        lm = self._LocalMinList
        while lm is not None:
            self._InsertScanbeam(lm.y)
            lm = lm.nextLm

    def Clear(self):
        self._PolyOutList = []
        ClipperBase.Clear(self)

    def _InsertScanbeam(self, y):
        if self._Scanbeam is None:
            self._Scanbeam = Scanbeam(y)
        elif y > self._Scanbeam.y:
            self._Scanbeam = Scanbeam(y, self._Scanbeam)
        else:
            sb = self._Scanbeam
            while sb.nextSb is not None and y <= sb.nextSb.y:
                sb = sb.nextSb
            if y == sb.y: return
            newSb = Scanbeam(y, sb.nextSb)
            sb.nextSb = newSb

    def _PopScanbeam(self):
        result = self._Scanbeam.y
        self._Scanbeam = self._Scanbeam.nextSb
        return result

    def _SetWindingCount(self, edge):
        e = edge.prevInAEL
        while e is not None and e.PolyType != edge.PolyType:
            e = e.prevInAEL
        if e is None:
            edge.windCnt = edge.windDelta
            edge.windCnt2 = 0
            e = self._ActiveEdges
        elif self._IsEvenOddFillType(edge):
            edge.windCnt = 1
            edge.windCnt2 = e.windCnt2
            e = e.nextInAEL
        else:
            if e.windCnt * e.windDelta < 0:
                if (abs(e.windCnt) > 1):
                    if (e.windDelta * edge.windDelta < 0): edge.windCnt = e.windCnt
                    else: edge.windCnt = e.windCnt + edge.windDelta
                else:
                    edge.windCnt = e.windCnt + e.windDelta + edge.windDelta
            elif (abs(e.windCnt) > 1) and (e.windDelta * edge.windDelta < 0):
                edge.windCnt = e.windCnt
            elif e.windCnt + edge.windDelta == 0:
                edge.windCnt = e.windCnt
            else:
                edge.windCnt = e.windCnt + edge.windDelta
            edge.windCnt2 = e.windCnt2
            e = e.nextInAEL
        # update windCnt2 ...
        if self._IsEvenOddAltFillType(edge):
            while (e != edge):
                if edge.windCnt2 == 0: edge.windCnt2 = 1
                else: edge.windCnt2 = 0
                e = e.nextInAEL
        else:
            while (e != edge):
                edge.windCnt2 += e.windDelta
                e = e.nextInAEL

    def _IsEvenOddFillType(self, edge):
        if edge.PolyType == PolyType.Subject:
            return self._SubjFillType == PolyFillType.EvenOdd
        else:
            return self._ClipFillType == PolyFillType.EvenOdd

    def _IsEvenOddAltFillType(self, edge):
        if edge.PolyType == PolyType.Subject:
            return self._ClipFillType == PolyFillType.EvenOdd
        else:
            return self._SubjFillType == PolyFillType.EvenOdd

    def _IsContributing(self, edge):
        if edge.PolyType == PolyType.Subject:
            pft = self._SubjFillType
            pft2 = self._ClipFillType
        else:
            pft = self._ClipFillType
            pft2 = self._SubjFillType
        if pft == PolyFillType.EvenOdd or pft == PolyFillType.NonZero:
            if abs(edge.windCnt) != 1: return False
        elif pft == PolyFillType.Positive:
            if edge.windCnt != 1: return False
        elif pft == PolyFillType.Negative:
            if edge.windCnt != -1: return False

        if self._ClipType == ClipType.Intersection: ###########
            if pft2 == PolyFillType.EvenOdd or pft2 == PolyFillType.NonZero:
                return edge.windCnt2 != 0
            elif pft2 == PolyFillType.Positive:
                return edge.windCnt2 > 0
            else:
                return edge.windCnt2 < 0 # Negative
        elif self._ClipType == ClipType.Union:      ###########
            if pft2 == PolyFillType.EvenOdd or pft2 == PolyFillType.NonZero:
                return edge.windCnt2 == 0
            elif pft2 == PolyFillType.Positive:
                return edge.windCnt2 <= 0
            else: return edge.windCnt2 >= 0 # Negative
        elif self._ClipType == ClipType.Difference: ###########
            if edge.PolyType == PolyType.Subject:
                if pft2 == PolyFillType.EvenOdd or pft2 == PolyFillType.NonZero:
                    return edge.windCnt2 == 0
                elif edge.PolyType == PolyFillType.Positive:
                    return edge.windCnt2 <= 0
                else:
                    return edge.windCnt2 >= 0
            else:                                   
                if pft2 == PolyFillType.EvenOdd or pft2 == PolyFillType.NonZero:
                    return edge.windCnt2 != 0
                elif pft2 == PolyFillType.Positive:
                    return edge.windCnt2 > 0
                else:
                    return edge.windCnt2 < 0
        else: # self._ClipType == ClipType.XOR:     ###########
            return True 

    def _AddEdgeToSEL(self, edge):
        if self._SortedEdges is None:
            self._SortedEdges = edge
            edge.prevInSEL = None
            edge.nextInSEL = None
        else:
            # add edge to front of list ...
            edge.nextInSEL = self._SortedEdges
            edge.prevInSEL = None
            self._SortedEdges.prevInSEL = edge
            self._SortedEdges = edge

    def _CopyAELToSEL(self):
        e = self._ActiveEdges
        self._SortedEdges = e
        while e is not None:
            e.prevInSEL = e.prevInAEL
            e.nextInSEL = e.nextInAEL
            e = e.nextInAEL

    def _InsertEdgeIntoAEL(self, edge):
        edge.prevInAEL = None
        edge.nextInAEL = None
        if self._ActiveEdges is None:
            self._ActiveEdges = edge
        elif _E2InsertsBeforeE1(self._ActiveEdges, edge):
            edge.nextInAEL = self._ActiveEdges
            self._ActiveEdges.prevInAEL = edge
            self._ActiveEdges = edge
        else:
            e = self._ActiveEdges
            while e.nextInAEL is not None and \
                not _E2InsertsBeforeE1(e.nextInAEL, edge):
                    e = e.nextInAEL
            edge.nextInAEL = e.nextInAEL
            if e.nextInAEL is not None: e.nextInAEL.prevInAEL = edge
            edge.prevInAEL = e
            e.nextInAEL = edge

    def _InsertLocalMinimaIntoAEL(self, botY):
        while self._CurrentLocMin is not None and \
                 self._CurrentLocMin.y == botY:
            lb = self._CurrentLocMin.leftBound
            rb = self._CurrentLocMin.rightBound
            self._InsertEdgeIntoAEL(lb)
            self._InsertScanbeam(lb.yTop)
            self._InsertEdgeIntoAEL(rb)
            if self._IsEvenOddFillType(lb):
                lb.windDelta = 1
                rb.windDelta = 1
            else:
                rb.windDelta = -lb.windDelta
            self._SetWindingCount(lb)
            rb.windCnt = lb.windCnt
            rb.windCnt2 = lb.windCnt2
            if rb.dx == horizontal:
                self._AddEdgeToSEL(rb)
                self._InsertScanbeam(rb.nextInLML.yTop)
            else:
                self._InsertScanbeam(rb.yTop)
            if self._IsContributing(lb):
                self._AddLocalMinPoly(lb, rb, Point(lb.xCurr, self._CurrentLocMin.y))
            
            if rb.outIdx >= 0 and rb.dx == horizontal and self._HorzJoins is not None:
                hj = self._HorzJoins
                while True:
                    dummy1, dummy2, overlap = _GetOverlapSegment(Point(hj.edge.xBot, hj.edge.yBot),
                                                 Point(hj.edge.xTop, hj.edge.yTop), 
                                                 Point(rb.xBot, rb.yBot),
                                                 Point(rb.xTop, rb.yTop))
                    if overlap:
                        self._AddJoin(hj.edge, rb, hj.savedIdx)
                    hj = hj.nextHj
                    if hj == self._HorzJoins: break
            
            if (lb.nextInAEL != rb):
                
                if rb.outIdx >= 0 and rb.prevInAEL.outIdx >= 0 and _SlopesEqual2(rb.prevInAEL, rb):
                    self._AddJoin(rb, rb.prevInAEL)
                
                e = lb.nextInAEL
                pt = Point(lb.xCurr, lb.yCurr)
                while e != rb:
                    self._IntersectEdges(rb, e, pt)
                    e = e.nextInAEL
            self._PopLocalMinima()

    def _SwapPositionsInAEL(self, e1, e2):
        if e1.nextInAEL == e2:
            nextE = e2.nextInAEL
            if nextE is not None: nextE.prevInAEL = e1
            prevE = e1.prevInAEL
            if prevE is not None: prevE.nextInAEL = e2
            e2.prevInAEL = prevE
            e2.nextInAEL = e1
            e1.prevInAEL = e2
            e1.nextInAEL = nextE
        elif e2.nextInAEL == e1:
            nextE = e1.nextInAEL
            if nextE is not None: nextE.prevInAEL = e2
            prevE = e2.prevInAEL
            if prevE is not None: prevE.nextInAEL = e1
            e1.prevInAEL = prevE
            e1.nextInAEL = e2
            e2.prevInAEL = e1
            e2.nextInAEL = nextE
        else:
            nextE = e1.nextInAEL
            prevE = e1.prevInAEL
            e1.nextInAEL = e2.nextInAEL
            if e1.nextInAEL is not None: e1.nextInAEL.prevInAEL = e1
            e1.prevInAEL = e2.prevInAEL
            if e1.prevInAEL is not None: e1.prevInAEL.nextInAEL = e1
            e2.nextInAEL = nextE
            if e2.nextInAEL is not None: e2.nextInAEL.prevInAEL = e2
            e2.prevInAEL = prevE
            if e2.prevInAEL is not None: e2.prevInAEL.nextInAEL = e2
        if e1.prevInAEL is None: self._ActiveEdges = e1
        elif e2.prevInAEL is None: self._ActiveEdges = e2

    def _SwapPositionsInSEL(self, e1, e2):
        if e1.nextInSEL == e2:
            nextE = e2.nextInSEL
            if nextE is not None: nextE.prevInSEL = e1
            prevE = e1.prevInSEL
            if prevE is not None: prevE.nextInSEL = e2
            e2.prevInSEL = prevE
            e2.nextInSEL = e1
            e1.prevInSEL = e2
            e1.nextInSEL = nextE
        elif e2.nextInSEL == e1:
            nextE = e1.nextInSEL
            if nextE is not None: nextE.prevInSEL = e2
            prevE = e2.prevInSEL
            if prevE is not None: prevE.nextInSEL = e1
            e1.prevInSEL = prevE
            e1.nextInSEL = e2
            e2.prevInSEL = e1
            e2.nextInSEL = nextE
        else:
            nextE = e1.nextInSEL
            prevE = e1.prevInSEL
            e1.nextInSEL = e2.nextInSEL
            e1.nextInSEL = e2.nextInSEL
            if e1.nextInSEL is not None: e1.nextInSEL.prevInSEL = e1
            e1.prevInSEL = e2.prevInSEL
            if e1.prevInSEL is not None: e1.prevInSEL.nextInSEL = e1
            e2.nextInSEL = nextE
            if e2.nextInSEL is not None: e2.nextInSEL.prevInSEL = e2
            e2.prevInSEL = prevE
            if e2.prevInSEL is not None: e2.prevInSEL.nextInSEL = e2
        if e1.prevInSEL is None: self._SortedEdges = e1
        elif e2.prevInSEL is None: self._SortedEdges = e2

    def _IsTopHorz(self, xPos):
        e = self._SortedEdges
        while e is not None:
            if (xPos >= min(e.xCurr,e.xTop)) and (xPos <= max(e.xCurr,e.xTop)):
                return False
            e = e.nextInSEL
        return True

    def _ProcessHorizontal(self, horzEdge):
        if horzEdge.xCurr < horzEdge.xTop:
            horzLeft = horzEdge.xCurr
            horzRight = horzEdge.xTop
            direction = Direction.LeftToRight
        else:
            horzLeft = horzEdge.xTop
            horzRight = horzEdge.xCurr
            direction = Direction.RightToLeft
        eMaxPair = None
        if horzEdge.nextInLML is None:
            eMaxPair = _GetMaximaPair(horzEdge)
        e = _GetnextInAEL(horzEdge, direction)
        while e is not None:
            if (e.xCurr == horzEdge.xTop) and eMaxPair is None:
                if _SlopesEqual2(e, horzEdge.nextInLML): 
                    if horzEdge.outIdx >= 0 and e.outIdx >= 0:
                        self._AddJoin(horzEdge.nextInLML, e, horzEdge.outIdx)
                    break
                elif e.dx < horzEdge.nextInLML.dx: break
            eNext = _GetnextInAEL(e, direction)
            if eMaxPair is not None or \
                ((direction == Direction.LeftToRight) and (e.xCurr < horzRight)) or \
                ((direction == Direction.RightToLeft) and (e.xCurr > horzLeft)):
                if e == eMaxPair:
                    if direction == Direction.LeftToRight:
                        self._IntersectEdges(horzEdge, e, Point(e.xCurr, horzEdge.yCurr))
                    else:
                        self._IntersectEdges(e, horzEdge, Point(e.xCurr, horzEdge.yCurr))
                    return
                elif e.dx == horizontal and not _IsMinima(e) and e.xCurr <= e.xTop:
                    if direction == Direction.LeftToRight:
                        self._IntersectEdges(horzEdge, e, Point(e.xCurr, horzEdge.yCurr),
                            _ProtectRight(not self._IsTopHorz(e.xCurr)))
                    else:
                        self._IntersectEdges(e, horzEdge, Point(e.xCurr, horzEdge.yCurr),
                            _ProtectLeft(not self._IsTopHorz(e.xCurr)))
                elif (direction == Direction.LeftToRight):
                    self._IntersectEdges(horzEdge, e, Point(e.xCurr, horzEdge.yCurr),
                        _ProtectRight(not self._IsTopHorz(e.xCurr)))
                else:
                    self._IntersectEdges(e, horzEdge, Point(e.xCurr, horzEdge.yCurr),
                        _ProtectLeft(not self._IsTopHorz(e.xCurr)))
                self._SwapPositionsInAEL(horzEdge, e)
            elif ((direction == Direction.LeftToRight and e.xCurr >= horzRight) or \
                (direction == Direction.RightToLeft and e.xCurr <= horzLeft)): break
            e = eNext
        if horzEdge.nextInLML is not None:
            if horzEdge.outIdx >= 0:
                self._AddOutPt(horzEdge, Point(horzEdge.xTop, horzEdge.yTop))
            self._UpdateEdgeIntoAEL(horzEdge)
        else:
            if horzEdge.outIdx >= 0:
                self._IntersectEdges(horzEdge, eMaxPair, \
                    Point(horzEdge.xTop, horzEdge.yCurr), Protects.Both)
            if eMaxPair.outIdx >= 0: raise Exception("Clipper: Horizontal Error")
            self._DeleteFromAEL(eMaxPair)
            self._DeleteFromAEL(horzEdge)

    def _ProcessHorizontals(self):
        while self._SortedEdges is not None:
            e = self._SortedEdges
            self._DeleteFromSEL(e)
            self._ProcessHorizontal(e)
            
    def _AddJoin(self, e1, e2, e1OutIdx = -1, e2OutIdx = -1):
        jr = JoinRec()
        if e1OutIdx >= 0: jr.poly1Idx = e1OutIdx
        else: jr.poly1Idx = e1.outIdx
        jr.pt1a = Point(e1.xCurr, e1.yCurr)
        jr.pt1b = Point(e1.xTop, e1.yTop)
        if e2OutIdx >= 0: jr.poly2Idx = e2OutIdx 
        else: jr.poly2Idx = e2.outIdx
        jr.pt2a = Point(e2.xCurr, e2.yCurr)
        jr.pt2b = Point(e2.xTop, e2.yTop)
        if self._JoinList is None: 
            self._JoinList = []
        self._JoinList.append(jr)
        
    def _FixupJoinRecs(self, jr, outPt, startIdx):
        for i in range(startIdx, len(self._JoinList)):
            jr2 = self._JoinList[i]
            if jr2.poly1Idx == jr.poly1Idx and _PointIsVertex(jr2.pt1a, outPt):
                jr2.poly1Idx = jr.poly2Idx
            if jr2.poly2Idx == jr.poly1Idx and _PointIsVertex(jr2.pt2a, outPt):
                jr2.poly2Idx = jr.poly2Idx
                
    def _AddHorzJoin(self, e, idx):
        hj = HorzJoin(e, idx)
        if self._HorzJoins == None:
            self._HorzJoins = hj
            hj.nextHj = hj
            hj.prevHj = hj
        else:
            hj.nextHj = self._HorzJoins
            hj.prevHj = self._HorzJoins.prevHj
            self._HorzJoins.prevHj.nextHj = hj
            self._HorzJoins.prevHj = hj

    def _InsertIntersectNode(self, e1, e2, pt):
        newNode = IntersectNode(e1, e2, pt)
        if self._IntersectNodes is None:
            self._IntersectNodes = newNode
        elif newNode.pt.y > self._IntersectNodes.pt.y:
            newNode.nextIn = self._IntersectNodes
            self._IntersectNodes = newNode
        else:
            node = self._IntersectNodes
            while node.nextIn is not None and \
                newNode.pt.y < node.nextIn.pt.y:
                node = node.nextIn
            newNode.nextIn = node.nextIn
            node.nextIn = newNode

    def _ProcessIntersections(self, botY, topY):
        try:
            self._BuildIntersectList(botY, topY)
            if self._IntersectNodes is None: return True
            if self._IntersectNodes.nextIn is not None and \
                not self._FixupIntersectionOrder(): return False 
            self._ProcessIntersectList()
            return True
        finally:
            self._IntersectNodes = None
            self._SortedEdges = None
            
    def _BuildIntersectList(self, botY, topY):
        e = self._ActiveEdges
        if e is None: return
        self._SortedEdges = e
        while e is not None:
            e.prevInSEL = e.prevInAEL
            e.nextInSEL = e.nextInAEL
            e.xCurr = _TopX(e, topY)
            e = e.nextInAEL
        while True:
            isModified = False
            e = self._SortedEdges
            while e.nextInSEL is not None:
                eNext = e.nextInSEL
                if e.xCurr <= eNext.xCurr:
                    e = eNext
                    continue
                pt, intersected = _IntersectPoint(e, eNext)
                if not intersected and e.xCurr > eNext.xCurr +1: 
                    raise Exception("Intersect Error")  
                if pt.y > botY:
                    pt = Point(_TopX(e, botY), botY)
                self._InsertIntersectNode(e, eNext, pt)
                self._SwapPositionsInSEL(e, eNext)
                isModified = True
            if e.prevInSEL is not None:
                e.prevInSEL.nextInSEL = None
            else:
                break
            if not isModified: break
        self._SortedEdges = None
        return

    def _ProcessIntersectList(self):
        while self._IntersectNodes is not None:
            node = self._IntersectNodes
            self._IntersectEdges(node.e1, node.e2, node.pt, Protects.Both)
            self._SwapPositionsInAEL(node.e1, node.e2)
            self._IntersectNodes = node.nextIn

    def _DeleteFromAEL(self, e):
        aelPrev = e.prevInAEL
        aelNext = e.nextInAEL
        if aelPrev is None and aelNext is None and e != self._ActiveEdges:
            return
        if aelPrev is not None:
            aelPrev.nextInAEL = aelNext
        else:
            self._ActiveEdges = aelNext
        if aelNext is not None:
            aelNext.prevInAEL = aelPrev
        e.nextInAEL = None
        e.prevInAEL = None

    def _DeleteFromSEL(self, e):
        SELPrev = e.prevInSEL
        SELNext = e.nextInSEL
        if SELPrev is None and SELNext is None and e != self._SortedEdges:
            return
        if SELPrev is not None:
            SELPrev.nextInSEL = SELNext
        else:
            self._SortedEdges = SELNext
        if SELNext is not None:
            SELNext.prevInSEL = SELPrev
        e.nextInSEL = None
        e.prevInSEL = None

    def _IntersectEdges(self, e1, e2, pt, protects = Protects.Neither):
        e1stops = protects & Protects.Left == 0 and \
                e1.nextInLML is None and \
                e1.xTop == pt.x and e1.yTop == pt.y
        e2stops = protects & Protects.Right == 0 and \
                e2.nextInLML is None and \
                e2.xTop == pt.x and e2.yTop == pt.y
        e1Contributing = e1.outIdx >= 0
        e2contributing = e2.outIdx >= 0

        if e1.PolyType == e2.PolyType:
            if self._IsEvenOddFillType(e1):
                e1Wc = e1.windCnt
                e1.windCnt = e2.windCnt
                e2.windCnt = e1Wc
            else:
                if e1.windCnt + e2.windDelta == 0: e1.windCnt = -e1.windCnt
                else: e1.windCnt += e2.windDelta
                if e2.windCnt - e1.windDelta == 0: e2.windCnt = -e2.windCnt
                else: e2.windCnt -= e1.windDelta
        else:
            if not self._IsEvenOddFillType(e2): e1.windCnt2 += e2.windDelta
            elif e1.windCnt2 == 0: e1.windCnt2 = 1
            else: e1.windCnt2 = 0
            if not self._IsEvenOddFillType(e1): e2.windCnt2 -= e1.windDelta
            elif e2.windCnt2 == 0: e2.windCnt2 = 1
            else: e2.windCnt2 = 0

        if e1.PolyType == PolyType.Subject:
            e1FillType = self._SubjFillType
            e1FillType2 = self._ClipFillType
        else:
            e1FillType = self._ClipFillType
            e1FillType2 = self._SubjFillType

        if e2.PolyType == PolyType.Subject:
            e2FillType = self._SubjFillType
            e2FillType2 = self._ClipFillType
        else:
            e2FillType = self._ClipFillType
            e2FillType2 = self._SubjFillType

        if e1FillType == PolyFillType.Positive: e1Wc = e1.windCnt
        elif e1FillType == PolyFillType.Negative: e1Wc = -e1.windCnt
        else: e1Wc = abs(e1.windCnt)

        if e2FillType == PolyFillType.Positive: e2Wc = e2.windCnt
        elif e2FillType == PolyFillType.Negative: e2Wc = -e2.windCnt
        else: e2Wc = abs(e2.windCnt)

        if e1Contributing and e2contributing:
            if e1stops or e2stops or \
                (e1Wc != 0 and e1Wc != 1) or (e2Wc != 0 and e2Wc != 1) or \
                (e1.PolyType != e2.PolyType and self._ClipType != ClipType.Xor):
                    self._AddLocalMaxPoly(e1, e2, pt)
            else:
                self._AddOutPt(e1, pt)
                self._AddOutPt(e2, pt)
                _SwapSides(e1, e2)
                _SwapPolyIndexes(e1, e2)
        elif e1Contributing:
            if (e2Wc == 0 or e2Wc == 1): 
                self._AddOutPt(e1, pt)
                _SwapSides(e1, e2)
                _SwapPolyIndexes(e1, e2)
        elif e2contributing:
            if (e1Wc == 0 or e1Wc == 1): 
                self._AddOutPt(e2, pt)
                _SwapSides(e1, e2)
                _SwapPolyIndexes(e1, e2)

        elif    (e1Wc == 0 or e1Wc == 1) and (e2Wc == 0 or e2Wc == 1) and \
            not e1stops and not e2stops:

            e1FillType2 = e2FillType2 = PolyFillType.EvenOdd
            if e1FillType2 == PolyFillType.Positive: e1Wc2 = e1.windCnt2
            elif e1FillType2 == PolyFillType.Negative: e1Wc2 = -e1.windCnt2
            else: e1Wc2 = abs(e1.windCnt2)
            if e2FillType2 == PolyFillType.Positive: e2Wc2 = e2.windCnt2
            elif e2FillType2 == PolyFillType.Negative: e2Wc2 = -e2.windCnt2
            else: e2Wc2 = abs(e2.windCnt2)

            if e1.PolyType != e2.PolyType:
                self._AddLocalMinPoly(e1, e2, pt)
            elif e1Wc == 1 and e2Wc == 1:
                if self._ClipType == ClipType.Intersection:
                    if e1Wc2 > 0 and e2Wc2 > 0:
                        self._AddLocalMinPoly(e1, e2, pt)
                elif self._ClipType == ClipType.Union:
                    if e1Wc2 <= 0 and e2Wc2 <= 0:
                        self._AddLocalMinPoly(e1, e2, pt)
                elif self._ClipType == ClipType.Difference:
                    if (e1.PolyType == PolyType.Clip and e1Wc2 > 0 and e2Wc2 > 0) or \
                        (e1.PolyType == PolyType.Subject and e1Wc2 <= 0 and e2Wc2 <= 0):
                            self._AddLocalMinPoly(e1, e2, pt)
                else:
                    self._AddLocalMinPoly(e1, e2, pt)
            else:
                _SwapSides(e1, e2, self._PolyOutList)

        if e1stops != e2stops and \
            ((e1stops and e1.outIdx >= 0) or (e2stops and e2.outIdx >= 0)):
                _SwapSides(e1, e2, self._PolyOutList)
                _SwapPolyIndexes(e1, e2)
        if e1stops: self._DeleteFromAEL(e1)
        if e2stops: self._DeleteFromAEL(e2)

    def _DoMaxima(self, e, topY):
        eMaxPair = _GetMaximaPair(e)
        x = e.xTop
        eNext = e.nextInAEL
        while eNext != eMaxPair:
            if eNext is None: raise Exception("DoMaxima error")
            self._IntersectEdges(e, eNext, Point(x, topY), Protects.Both)
            self._SwapPositionsInAEL(e, eNext)
            eNext = e.nextInAEL
        if e.outIdx < 0 and eMaxPair.outIdx < 0:
            self._DeleteFromAEL(e)
            self._DeleteFromAEL(eMaxPair)
        elif e.outIdx >= 0 and eMaxPair.outIdx >= 0:
            self._IntersectEdges(e, eMaxPair, Point(x, topY))
        else:
            raise Exception("DoMaxima error")

    def _UpdateEdgeIntoAEL(self, e):
        if e.nextInLML is None:
            raise Exception("UpdateEdgeIntoAEL error")
        aelPrev = e.prevInAEL
        aelNext = e.nextInAEL
        e.nextInLML.outIdx = e.outIdx
        if aelPrev is not None:
            aelPrev.nextInAEL = e.nextInLML
        else:
            self._ActiveEdges = e.nextInLML
        if aelNext is not None:
            aelNext.prevInAEL = e.nextInLML
        e.nextInLML.side = e.side
        e.nextInLML.windDelta = e.windDelta
        e.nextInLML.windCnt = e.windCnt
        e.nextInLML.windCnt2 = e.windCnt2
        e = e.nextInLML
        e.prevInAEL = aelPrev
        e.nextInAEL = aelNext
        if e.dx != horizontal:
            self._InsertScanbeam(e.yTop)
        return e

    def _AddLocalMinPoly(self, e1, e2, pt):
        if e2.dx == horizontal or e1.dx > e2.dx:
            self._AddOutPt(e1, pt)
            e2.outIdx = e1.outIdx
            e1.side = EdgeSide.Left
            e2.side = EdgeSide.Right
            e = e1
            if e.prevInAEL == e2: prevE = e2.prevInAEL
            else: prevE = e1.prevInAEL
        else:
            self._AddOutPt(e2, pt)
            e1.outIdx = e2.outIdx
            e1.side = EdgeSide.Right
            e2.side = EdgeSide.Left
            e = e2
            if e.prevInAEL == e1: prevE = e1.prevInAEL
            else: prevE = e.prevInAEL

        if prevE is not None and prevE.outIdx >= 0 and \
            _TopX(prevE, pt.y) == _TopX(e, pt.y) and \
           _SlopesEqual2(e, prevE): 
                self._AddJoin(e, prevE)
        return

    def _AddLocalMaxPoly(self, e1, e2, pt):
        self._AddOutPt(e1, pt)
        if e1.outIdx == e2.outIdx:
            e1.outIdx = -1
            e2.outIdx = -1
        elif e1.outIdx < e2.outIdx:
            self._AppendPolygon(e1, e2)
        else:
            self._AppendPolygon(e2, e1)

    def _CreateOutRec(self):
        outRec = OutRec(len(self._PolyOutList))
        self._PolyOutList.append(outRec)
        return outRec
    
    def _AddOutPt(self, e, pt):
        toFront = e.side == EdgeSide.Left
        if e.outIdx < 0:
            outRec = self._CreateOutRec();
            e.outIdx = outRec.idx
            op = OutPt(outRec.idx, pt)
            op.nextOp = op
            op.prevOp = op
            outRec.pts = op
            _SetHoleState(e, outRec, self._PolyOutList)
        else:
            outRec = self._PolyOutList[e.outIdx]
            op = outRec.pts
            if (toFront and _PointsEqual(pt, op.pt)) or \
                (not toFront and _PointsEqual(pt, op.prevOp.pt)): return
            op2 = OutPt(outRec.idx, pt)
            op2.nextOp = op
            op2.prevOp = op.prevOp
            op.prevOp.nextOp = op2
            op.prevOp = op2
            if toFront: outRec.pts = op2
        
    def _AppendPolygon(self, e1, e2):
        outRec1 = self._PolyOutList[e1.outIdx]
        outRec2 = self._PolyOutList[e2.outIdx]
        holeStateRec = None
        if _Param1RightOfParam2(outRec1, outRec2): holeStateRec = outRec2
        elif _Param1RightOfParam2(outRec2, outRec1): holeStateRec = outRec1
        else: holeStateRec = _GetLowermostRec(outRec1, outRec2)
                
        p1_lft = outRec1.pts
        p2_lft = outRec2.pts
        p1_rt = p1_lft.prevOp
        p2_rt = p2_lft.prevOp
        newSide = EdgeSide.Left
        
        if e1.side == EdgeSide.Left:
            if e2.side == EdgeSide.Left:
                # z y x a b c
                _ReversePolyPtLinks(p2_lft)
                p2_lft.nextOp = p1_lft
                p1_lft.prevOp = p2_lft
                p1_rt.nextOp = p2_rt
                p2_rt.prevOp = p1_rt
                outRec1.pts = p2_rt
            else:
                # x y z a b c
                p2_rt.nextOp = p1_lft
                p1_lft.prevOp = p2_rt
                p2_lft.prevOp = p1_rt
                p1_rt.nextOp = p2_lft
                outRec1.pts = p2_lft
        else:
            newSide = EdgeSide.Right
            if e2.side == EdgeSide.Right:
                # a b c z y x
                _ReversePolyPtLinks(p2_lft)
                p1_rt.nextOp = p2_rt
                p2_rt.prevOp = p1_rt
                p2_lft.nextOp = p1_lft
                p1_lft.prevOp = p2_lft
            else:
                # a b c x y z
                p1_rt.nextOp = p2_lft
                p2_lft.prevOp = p1_rt
                p1_lft.prevOp = p2_rt
                p2_rt.nextOp = p1_lft
                
        outRec1.bottomPt = None                
        if holeStateRec == outRec2:
            if outRec2.FirstLeft != outRec1:
                outRec1.FirstLeft = outRec2.FirstLeft
            outRec1.isHole = outRec2.isHole
        outRec2.pts = None
        outRec2.bottomPt = None
        outRec2.FirstLeft = outRec1
        OKIdx = outRec1.idx
        ObsoleteIdx = outRec2.idx

        e1.outIdx = -1
        e2.outIdx = -1

        e = self._ActiveEdges
        while e is not None:
            if e.outIdx == ObsoleteIdx:
                e.outIdx = OKIdx
                e.side = newSide
                break
            e = e.nextInAEL
        outRec2.idx = outRec1.idx    
        
    def _FixupIntersectionOrder(self):
        self._CopyAELToSEL()
        inode = self._IntersectNodes
        while inode is not None:
            if (not _EdgesAdjacent(inode)):
                nextNode = inode.nextIn
                while (nextNode and not _EdgesAdjacent(nextNode)):
                    nextNode = nextNode.nextIn
                if (nextNode is None): return False
                e1 = inode.e1
                e2 = inode.e2
                p = inode.pt
                inode.e1 = nextNode.e1
                inode.e2 = nextNode.e2
                inode.pt = nextNode.pt
                nextNode.e1 = e1
                nextNode.e2 = e2
                nextNode.pt = p
        
            self._SwapPositionsInSEL(inode.e1, inode.e2);
            inode = inode.nextIn
        return True
                                
    def _ProcessEdgesAtTopOfScanbeam(self, topY):
        e = self._ActiveEdges
        while e is not None:
            if _IsMaxima(e, topY) and _GetMaximaPair(e).dx != horizontal:
                ePrev = e.prevInAEL
                self._DoMaxima(e, topY)
                if ePrev is None: e = self._ActiveEdges
                else: e = ePrev.nextInAEL
            else:
                intermediateVert = _IsIntermediate(e, topY)
                if intermediateVert and e.nextInLML.dx == horizontal:
                    if e.outIdx >= 0:
                        self._AddOutPt(e, Point(e.xTop, e.yTop))
                        hj = self._HorzJoins
                        if hj is not None:
                            while True:
                                _1, _2, overlap = _GetOverlapSegment(
                                                        Point(hj.edge.xBot, hj.edge.yBot),
                                                        Point(hj.edge.xTop, hj.edge.yTop),
                                                        Point(e.nextInLML.XBot, e.nextInLML.yBot),
                                                        Point(e.nextInLML.xTop, e.nextInLML.yTop))
                                if overlap: self._AddJoin(hj.edge, e.nextInLML, hj.savedIdx, e.outIdx)
                                hj = hj.nextHj
                            if hj == self._HorzJoins: break
                            self._AddHorzJoin(e.nextInLML, e.outIdx)                        
                        
                    e = self._UpdateEdgeIntoAEL(e)
                    self._AddEdgeToSEL(e)
                else:
                    e.xCurr = _TopX(e, topY)
                    e.yCurr = topY
                    if (self.ForceSimple and e.prevInAEL is not None and
                      e.prevInAEL.xCurr == e.xCurr and
                      e.outIdx >= 0 and e.prevInAEL.outIdx >= 0):
                        if (intermediateVert):
                            self._AddOutPt(e.prevInAEL, Point(e.xCurr, topY));
                        else:
                            self._AddOutPt(e, Point(e.xCurr, topY))
                e = e.nextInAEL

        self._ProcessHorizontals()

        e = self._ActiveEdges
        while e is not None:
            if _IsIntermediate(e, topY):
                if (e.outIdx >= 0) :
                    self._AddOutPt(e, Point(e.xTop, e.yTop))
                e = self._UpdateEdgeIntoAEL(e)
                
                ePrev = e.prevInAEL
                eNext  = e.nextInAEL
                if ePrev is not None and ePrev.xCurr == e.xBot and \
                    (ePrev.yCurr == e.yBot) and (e.outIdx >= 0) and \
                    (ePrev.outIdx >= 0) and (ePrev.yCurr > ePrev.yTop) and \
                    _SlopesEqual2(e, ePrev):
                        self._AddOutPt(ePrev, Point(e.xBot, e.yBot))
                        self._AddJoin(e, ePrev)
                elif eNext is not None and (eNext.xCurr == e.xBot) and \
                    (eNext.yCurr == e.yBot) and (e.outIdx >= 0) and \
                    (eNext.outIdx >= 0) and (eNext.yCurr > eNext.yTop) and \
                    _SlopesEqual2(e, eNext):
                        self._AddOutPt(eNext, Point(e.xBot, e.yBot))
                        self._AddJoin(e, eNext)
                
            e = e.nextInAEL
                      
    def _Area(self, pts):
        # see http://www.mathopenref.com/coordpolygonarea2.html
        result = 0.0
        p = pts
        while True:
            result += (p.pt.x + p.prevOp.pt.x) * (p.prevOp.pt.y - p.pt.y)
            p = p.nextOp
            if p == pts: break
        return result / 2
        
    def _JoinPoints(self, jr):
        p1, p2 = None, None
        outRec1 = self._PolyOutList[jr.poly1Idx]
        outRec2 = self._PolyOutList[jr.poly2Idx]
        if outRec1 is None or outRec2 is None: return p1, p2, False        
        pp1a = outRec1.pts; pp2a = outRec2.pts
        pt1 = jr.pt2a; pt2 = jr.pt2b
        pt3 = jr.pt1a; pt4 = jr.pt1b
        pp1a, pt1, pt2, result = _FindSegment(pp1a, pt1, pt2)
        if not result: return p1, p2, False
        if (outRec1 == outRec2):
            pp2a = pp1a.nextOp
            pp2a, pt3, pt4, result = _FindSegment(pp2a, pt3, pt4) 
            if not result or pp2a == pp1a: return p1, p2, False
        else:
            pp2a, pt3, pt4, result = _FindSegment(pp2a, pt3, pt4)
            if not result: return p1, p2, False
        pt1, pt2, result = _GetOverlapSegment(pt1, pt2, pt3, pt4) 
        if not result: return p1, p2, False
    
        prevOp = pp1a.prevOp
        if _PointsEqual(pp1a.pt, pt1): p1 = pp1a
        elif _PointsEqual(prevOp.pt, pt1): p1 = prevOp
        else: p1 = _InsertPolyPtBetween(pp1a, prevOp, pt1)
        
        if _PointsEqual(pp1a.pt, pt2): p2 = pp1a
        elif _PointsEqual(prevOp.pt, pt2): p2 = prevOp
        elif (p1 == pp1a) or (p1 == prevOp):
            p2 = _InsertPolyPtBetween(pp1a, prevOp, pt2)
        elif _Pt3IsBetweenPt1AndPt2(pp1a.pt, p1.pt, pt2):
            p2 = _InsertPolyPtBetween(pp1a, p1, pt2)
        else: p2 = _InsertPolyPtBetween(p1, prevOp, pt2)
    
        prevOp = pp2a.prevOp
        if _PointsEqual(pp2a.pt, pt1): p3 = pp2a
        elif _PointsEqual(prevOp.pt, pt1): p3 = prevOp
        else: p3 = _InsertPolyPtBetween(pp2a, prevOp, pt1)        
        if _PointsEqual(pp2a.pt, pt2): p4 = pp2a
        elif _PointsEqual(prevOp.pt, pt2): p4 = prevOp
        elif (p3 == pp2a) or (p3 == prevOp):
            p4 = _InsertPolyPtBetween(pp2a, prevOp, pt2)
        elif _Pt3IsBetweenPt1AndPt2(pp2a.pt, p3.pt, pt2):
            p4 = _InsertPolyPtBetween(pp2a, p3, pt2)
        else: p4 = _InsertPolyPtBetween(p3, prevOp, pt2)
    
        if p1.nextOp == p2 and p3.prevOp == p4:
            p1.nextOp = p3
            p3.prevOp = p1
            p2.prevOp = p4
            p4.nextOp = p2
            return p1, p2, True
        elif p1.prevOp == p2 and p3.nextOp == p4:
            p1.prevOp = p3
            p3.nextOp = p1
            p2.nextOp = p4
            p4.prevOp = p2
            return p1, p2, True
        return p1, p2, False

    def _FixupFirstLefts1(self, oldOutRec, newOutRec):
        for outRec in self._PolyOutList:
            if outRec.pts is not None and outRec.FirstLeft == oldOutRec:
                if _Poly2ContainsPoly1(outRec.pts, newOutRec.pts):
                    outRec.FirstLeft = newOutRec

    def _FixupFirstLefts2(self, oldOutRec, newOutRec):
        for outRec in self._PolyOutList:
            if outRec.FirstLeft == oldOutRec: outRec.FirstLeft = newOutRec

    def _GetOutRec(self, idx):
        outrec = self._PolyOutList[idx]
        while (outrec != self._PolyOutList[outrec.idx]):
            outrec = self._PolyOutList[outrec.idx]
        return outrec

    def _JoinCommonEdges(self):
        for i in range(len(self._JoinList)):
            jr = self._JoinList[i]
            outRec1 = self._GetOutRec(jr.poly1Idx)
            outRec2 = self._GetOutRec(jr.poly2Idx)
            if outRec1.pts is None or outRec2.pts is None: continue

            if outRec1 == outRec2: holeStateRec = outRec1
            elif _Param1RightOfParam2(outRec1, outRec2): holeStateRec = outRec2
            elif _Param1RightOfParam2(outRec2, outRec1): holeStateRec = outRec1
            else: holeStateRec = _GetLowermostRec(outRec1, outRec2)

            p1, p2, result = self._JoinPoints(jr)
            if not result: continue

            if outRec1 == outRec2:
                outRec1.pts = p1
                outRec1.bottomPt = None
                outRec2 = self._CreateOutRec()
                outRec2.pts = p2
                jr.poly2Idx = outRec2.idx

                if _Poly2ContainsPoly1(outRec2.pts, outRec1.pts):
                    outRec2.isHole = not outRec1.isHole
                    outRec2.FirstLeft = outRec1
                    
                    self._FixupJoinRecs(jr, p2, i + 1)
                    
                    if self._UsingPolyTree: self._FixupFirstLefts2(outRec2, outRec1)
                    
                    _FixupOutPolygon(outRec1)
                    _FixupOutPolygon(outRec2)
                    
                    if outRec2.isHole == self._Area(outRec2) > 0.0:
                        _ReversePolyPtLinks(outRec2.pts)
                        
                elif _Poly2ContainsPoly1(outRec1.pts, outRec2.pts):
                    outRec2.isHole = outRec1.isHole
                    outRec1.isHole = not outRec2.isHole
                    outRec2.FirstLeft = outRec1.FirstLeft
                    outRec1.FirstLeft = outRec2
                    
                    self._FixupJoinRecs(jr, p2, i + 1)
                    
                    if self._UsingPolyTree: self._FixupFirstLefts2(outRec1, outRec2)
                    
                    _FixupOutPolygon(outRec1)
                    _FixupOutPolygon(outRec2)
                    
                    if outRec1.isHole == self._Area(outRec1) > 0.0:
                        _ReversePolyPtLinks(outRec1.pts)
                else:                  
                    outRec2.isHole = outRec1.isHole
                    outRec2.FirstLeft = outRec1.FirstLeft
                    
                    self._FixupJoinRecs(jr, p2, i + 1)
                    if self._UsingPolyTree: self._FixupFirstLefts1(outRec1, outRec2)
                    
                    _FixupOutPolygon(outRec1)
                    _FixupOutPolygon(outRec2)
            else:
                _FixupOutPolygon(outRec1)
                outRec2.pts = None
                outRec2.bottomPt = None
                outRec2.idx = outRec1.idx
                
                outRec1.isHole = holeStateRec.isHole
                if holeStateRec == outRec2:
                    outRec1.FirstLeft = outRec2.FirstLeft
                outRec2.FirstLeft = outRec1
                
                if self._UsingPolyTree: self._FixupFirstLefts2(outRec2, outRec1)
        return
    
    def _DoSimplePolygons(self):
        i = 0;
        while i < len(self._PolyOutList):
            outrec = self._PolyOutList[i]
            i +=1
            op = outrec.pts
            if (op is None): continue
            while True:
                op2 = op.nextOp
                while (op2 != outrec.pts): 
                    if (_PointsEqual(op.pt, op2.pt) and op2.nextOp != op and op2.prevOp != op): 
                        #split the polygon into two ...
                        op3 = op.prevOp
                        op4 = op2.prevOp
                        op.prevOp = op4
                        op4.nextOp = op
                        op2.prevOp = op3
                        op3.nextOp = op2
                        
                        outrec.pts = op
                        outrec2 = self._CreateOutRec();
                        outrec2.pts = op2;
                        _UpdateOutPtIdxs(outrec2)
                        if (_Poly2ContainsPoly1(outrec2.pts, outrec.pts)):
                            #OutRec2 is contained by OutRec1 ...
                            outrec2.isHole = not outrec.isHole
                            outrec2.FirstLeft = outrec
                      
                        elif (_Poly2ContainsPoly1(outrec.pts, outrec2.pts)):
                            #OutRec1 is contained by OutRec2 ...
                            outrec2.isHole = outrec.isHole
                            outrec.isHole = not outrec2.isHole
                            outrec2.FirstLeft = outrec.FirstLeft
                            outrec.FirstLeft = outrec2
                        else:
                            #the 2 polygons are separate ...
                            outrec2.isHole = outrec.isHole;
                            outrec2.FirstLeft = outrec.FirstLeft;
                        op2 = op; # ie get ready for the next iteration
                    op2 = op2.nextOp
                op = op.nextOp
                if op == outrec.pts: break
        return
                
    def _ExecuteInternal(self):
        try: 
            try:
                self._Reset()
                if self._Scanbeam is None: return True
                botY = self._PopScanbeam()
                while True:
                    self._InsertLocalMinimaIntoAEL(botY)
                    self._HorzJoins = None
                    self._ProcessHorizontals()
                    topY = self._PopScanbeam()
                    if not self._ProcessIntersections(botY, topY): return False
                    self._ProcessEdgesAtTopOfScanbeam(topY)
                    botY = topY
                    if self._Scanbeam is None and self._CurrentLocMin is None: break
                    
                for outRec in self._PolyOutList:
                    if outRec.pts is None: continue                
                    _FixupOutPolygon(outRec)
                    if outRec.pts is None: continue
                    if outRec.isHole == (self._Area(outRec.pts) > 0.0):
                        _ReversePolyPtLinks(outRec.pts)
                
                if self._JoinList is not None: self._JoinCommonEdges()
                if self.ForceSimple: self._DoSimplePolygons()
                
                return True
            finally:
                self._JoinList = None
                self._HorzJoins = None
        except:
            return False

    def Execute(
            self,
            clipType,
            solution,
            subjFillType = PolyFillType.EvenOdd,
            clipFillType = PolyFillType.EvenOdd):
        if self._ExecuteLocked: return False
        try:
            self._ExecuteLocked = True
            self._UsingPolyTree = True
            del solution[:]
            self._SubjFillType = subjFillType
            self._ClipFillType = clipFillType
            self._ClipType = clipType
            result = self._ExecuteInternal()
            if result: self._BuildResult(solution)
        finally:
            self._ExecuteLocked = False
            self._UsingPolyTree = False
        return result

    def Execute2(
            self,
            clipType,
            solutionTree,
            subjFillType = PolyFillType.EvenOdd,
            clipFillType = PolyFillType.EvenOdd):
        if self._ExecuteLocked: return False
        try:
            self._ExecuteLocked = True
            self._UsingPolyTree = True
            solutionTree.Clear()
            self._SubjFillType = subjFillType
            self._ClipFillType = clipFillType
            self._ClipType = clipType
            result = self._ExecuteInternal()
            if result: self._BuildResult2(solutionTree)
        finally:
            self._ExecuteLocked = False
            self._UsingPolyTree = False
        return result

    def _BuildResult(self, polygons):
        for outRec in self._PolyOutList:
            if outRec is None: continue
            cnt = _PointCount(outRec.pts)
            if (cnt < 3): continue
            poly = []
            op = outRec.pts
            for _ in range(cnt):
                poly.append(Point(op.pt.x, op.pt.y))
                op = op.prevOp
            polygons.append(poly)
        return
    
    def _BuildResult2(self, polyTree):
        for outRec in self._PolyOutList:
            if outRec is None: continue
            cnt = _PointCount(outRec.pts)
            if (cnt < 3): continue
            _FixHoleLinkage(outRec)
            
            # add nodes to _AllNodes list ...
            polyNode = PolyNode()
            polyTree._AllNodes.append(polyNode)
            outRec.PolyNode = polyNode
            op = outRec.pts
            while True:
                polyNode.Contour.append(op.pt)
                op = op.prevOp
                if op == outRec.pts: break
        # build the tree ...
        for outRec in self._PolyOutList:
            if outRec.PolyNode is None: continue
            if outRec.FirstLeft is None:
                polyTree._AddChild(outRec.PolyNode)
            else:
                outRec.FirstLeft.PolyNode._AddChild(outRec.PolyNode)                 
        return
       
#===============================================================================
# OffsetPolygons (+ ancilliary functions)
#===============================================================================

FloatPoint = namedtuple('FloatPoint', 'x y')
Rect = namedtuple('FloatPoint', 'left top right bottom')

def _GetUnitNormal(pt1, pt2):
    if pt2.x == pt1.x and pt2.y == pt1.y:
        return FloatPoint(0.0, 0.0)
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    f = 1.0 / math.hypot(dx, dy)
    dx = float(dx) * f
    dy = float(dy) * f
    return FloatPoint(dy, -dx)

def _BuildArc(pt, a1, a2, r, limit):
    arcFrac = abs(a2 - a1) / (2 * math.pi);
    steps = int(arcFrac * math.pi / math.acos(1 - limit / abs(r)))
    if steps < 2: steps = 2
    elif steps > 222.0 * arcFrac:
        steps = int(222.0 * arcFrac)
    
    result = []
    y = math.sin(a1)
    x = math.cos(a1)
    s = math.sin((a2-a1)/steps)
    c = math.cos((a2-a1)/steps)
    for _ in range(steps+1):
        result.append(FloatPoint(pt.x + round(x * r), pt.y + round(y * r)))
        x2 = x
        x = x * c - s * y    # cross product & dot product here ...
        y = x2 * s + y * c   # avoids repeat calls to the much slower sin() & cos()
        
    return result

def _GetBounds(pts):
    left = None
    for poly in pts:
        for pt in poly:
            left = pt.x
            top = pt.y
            right = pt.x
            bottom = pt.y
            break
        break
    
    for poly in pts:
        for pt in poly:
            if pt.x < left: left = pt.x
            if pt.x > right: right = pt.x
            if pt.y < top: top = pt.y
            if pt.y > bottom: bottom = pt.y
    if left is None: return Rect(0, 0, 0, 0)
    else: return Rect(left, top, right, bottom)

def _GetLowestPt(poly):
    # precondition: poly must not be empty
    result = poly[0]
    for pt in poly:
        if pt.y > result.y or (pt.y == result.y and pt.x < result.x):
            result = pt
    return result

def _StripDupPts(poly):
    if poly == []: return poly
    for i in range(1, len(poly)):
        if _PointsEqual(poly[i-1], poly[i]): poly.pop(i)
    i = len(poly) -1
    while i > 0 and _PointsEqual(poly[i], poly[0]):
        poly.pop(i)
        i -= 1
    return poly

def _OffsetInternal(polys, isPolygon, delta, jointype = JoinType.Square, endtype = EndType.Square, limit = 0.0): 
    
    def _DoSquare(pt):
        pt1 = Point(round(pt.x + Normals[k].x * delta), round(pt.y + Normals[k].y * delta))
        pt2 = Point(round(pt.x + Normals[j].x * delta), round(pt.y + Normals[j].y * delta))
        if (Normals[k].x*Normals[j].y-Normals[j].x*Normals[k].y) * delta >= 0:
            a1 = math.atan2(Normals[k].y, Normals[k].x)
            a2 = math.atan2(-Normals[j].y, -Normals[j].x)
            a1 = abs(a2 - a1);
            if a1 > math.pi: a1 = math.pi * 2 - a1
            dx = math.tan((math.pi - a1)/4) * abs(delta)
            
            pt1 = Point(round(pt1.x -Normals[k].y * dx), round(pt1.y + Normals[k].x * dx))
            result.append(pt1)
            pt2 = Point(round(pt2.x + Normals[j].y * dx), round(pt2.y - Normals[j].x * dx))
            result.append(pt2)
        else:
            result.append(pt1)
            result.append(pt)
            result.append(pt2)

    def _DoMiter(pt, r):
        if ((Normals[k].x* Normals[j].y - Normals[j].x * Normals[k].y) * delta >= 0):
            q = delta / r;
            result.append(Point(round(pt.x + (Normals[k].x + Normals[j].x) *q),
              round(pt.y + (Normals[k].y + Normals[j].y) *q)))
        else:
            pt1 = Point(round(pt.x + Normals[k].x * delta), \
                        round(pt.y + Normals[k].y * delta))
            pt2 = Point(round(pt.x + Normals[j].x * delta), \
                        round(pt.y + Normals[j].y * delta))
            result.append(pt1)
            result.append(pt)
            result.append(pt2)

    def _DoRound(pt, limit):
        pt1 = Point(round(pt.x + Normals[k].x * delta), \
                    round(pt.y + Normals[k].y * delta))
        pt2 = Point(round(pt.x + Normals[j].x * delta), \
                    round(pt.y + Normals[j].y * delta))
        result.append(pt1)
        if (Normals[k].x * Normals[j].y - Normals[j].x * Normals[k].y) *delta >= 0:
            if (Normals[j].x * Normals[k].x + Normals[j].y * Normals[k].y) < 0.985:
                a1 = math.atan2(Normals[k].y, Normals[k].x)
                a2 = math.atan2(Normals[j].y, Normals[j].x)
                if (delta > 0) and (a2 < a1): a2 = a2 + math.pi * 2
                elif (delta < 0) and (a2 > a1): a2 = a2 - math.pi * 2
                arc = _BuildArc(pt, a1, a2, delta, limit)
                result.extend(arc)
        else:
            result.append(pt)
        result.append(pt2)
        
    def _OffsetPoint(jointype, limit):
        if jointype == JoinType.Miter:
            r = 1.0 + (Normals[j].x * Normals[k].x + Normals[j].y * Normals[k].y)
            if (r >= rmin): _DoMiter(pts[j], r) 
            else: _DoSquare(pts[j])
        elif jointype == JoinType.Square: _DoSquare(pts[j])
        else: _DoRound(pts[j], limit)
        return j

    if delta == 0: return polys
    rmin = 0.5    
    if (jointype == JoinType.Miter):  
        if (limit > 2): 
            rmin = 2.0 / (limit * limit)
        limit = 0.25; #just in case endtype == EndType.Round
    else:
        if (limit <= 0): limit = 0.25
        elif (limit > abs(delta)): limit = abs(delta)
            
    res = []
    ppts = polys[:]    
    for pts in ppts:    
        Normals = []
        result = []
        cnt = len(pts)
        
        if (cnt == 0 or cnt < 3 and delta <= 0): continue
        elif (cnt == 1):
            res.append(_BuildArc(pts[0], 0, 2 * math.pi, delta, limit))
            continue
        
        forceClose = _PointsEqual(pts[0], pts[cnt - 1])
        if (forceClose): cnt -=1
        
        for j in range(cnt -1):
            Normals.append(_GetUnitNormal(pts[j], pts[j+1]))
        if isPolygon or forceClose: 
            Normals.append(_GetUnitNormal(pts[cnt-1], pts[0]))
        else:
            Normals.append(Normals[cnt-2])
    
    
        if (isPolygon or forceClose):
            k = cnt - 1
            for j in range(cnt):
                k = _OffsetPoint(jointype, limit)
            res.append(result)
                    
            if not isPolygon:
                result = []
                delta = -delta
                k = cnt - 1
                for j in range(cnt):
                    k = _OffsetPoint(jointype, limit)
                delta = -delta
                res.append(result[::-1])        
    
        else: 
            # offset the polyline going forward ...
            k = 0;
            for j in range(1, cnt-1):
                k = _OffsetPoint(jointype, limit)
            
            # handle the end (butt, round or square) ...
            if (endtype == EndType.Butt):
                j = cnt - 1
                pt1 = Point(round(float(pts[j].x) + Normals[j].x * delta), \
                    round(float(pts[j].y) + Normals[j].y * delta))
                result.append(pt1)
                pt1 = Point(round(float(pts[j].x) - Normals[j].x * delta), \
                    round(float(pts[j].y) - Normals[j].y * delta))
                result.append(pt1)
            else:
                j = cnt - 1;
                k = cnt - 2;
                Normals[j] = DoublePoint(-Normals[j].x, -Normals[j].y)
                if (endtype == EndType.Square): _DoSquare(pts[j])
                else: _DoRound(pts[j], limit)
            
            # re-build Normals ...
            for j in range(cnt -1, 0, -1):
                Normals[j] = DoublePoint(-Normals[j -1].x, -Normals[j -1].y)
            Normals[0] = DoublePoint(-Normals[1].x, -Normals[1].y)
            
            # offset the polyline going backward ...
            k = cnt -1;
            for j in range(cnt -2, 0, -1):
                k = _OffsetPoint(jointype, limit)
            
            # finally handle the start (butt, round or square) ...
            if (endtype == EndType.Butt): 
                pt1 = Point(round(float(pts[0].x) - Normals[0].x * delta), \
                    round(float(pts[0].y) - Normals[0].y * delta))
                result.append(pt1)
                pt1 = Point(round(float(pts[0].x) + Normals[0].x * delta), \
                    round(float(pts[0].y) + Normals[0].y * delta))
                result.append(pt1)
            else:
                j = 0
                k = 1
                if (endtype == EndType.Square): _DoSquare(pts[0]) 
                else: _DoRound(pts[0], limit)
            res.append(result)        
            

    c = Clipper()
    c.AddPolygons(res, PolyType.Subject)
    if delta > 0:
        c.Execute(ClipType.Union, res, PolyFillType.Positive, PolyFillType.Positive)
    else:
        bounds = _GetBounds(res)
        outer = []
        outer.append(Point(bounds.left-10, bounds.bottom+10))
        outer.append(Point(bounds.right+10, bounds.bottom+10))
        outer.append(Point(bounds.right+10, bounds.top-10))
        outer.append(Point(bounds.left-10, bounds.top-10))
        c.AddPolygon(outer, PolyType.Subject)
        c.Execute(ClipType.Union, res, PolyFillType.Negative, PolyFillType.Negative)
        if len(res) > 0: res.pop(0)
        for poly in res:
            poly = poly[::-1]             
    return res

def OffsetPolygons(polys, delta, jointype = JoinType.Square, limit = 0.0, autoFix = True):
    if not autoFix: 
        return _OffsetInternal(polys, True, delta, jointype, EndType.Butt, limit)        
    pts = polys[:]
    botPoly = None
    botPt = None
    for poly in pts:
        poly = _StripDupPts(poly)
        if len(poly) < 3: continue
        bot = _GetLowestPt(poly)
        if botPt is None or (bot.y > botPt.y) or \
            (bot.y == botPt.y and bot.x < botPt.x):
                botPt = bot
                botPoly = poly
    if botPt is None: return []
    # if the outermost polygon has the wrong orientation,
    # reverse the orientation of all the polygons ...
    if Area(botPoly) < 0.0:
        for i in range(len(pts)):
            pts[i] = pts[i][::-1]                
    return _OffsetInternal(pts, True, delta, jointype, EndType.Butt, limit)

def OffsetPolyLines(polys, delta, jointype = JoinType.Square, endtype = EndType.Square, limit = 0.0):
    polys2 = polys[:]
    for p in polys2:
        if p == []: continue            
        for i in range(1, len(p)):
            if _PointsEqual(p[i-1], p[i]): p.pop(i)

    if endtype == EndType.Closed:
        for i in range(len(polys2)):
            polys2.append(polys2[i][::-1])
        return _OffsetInternal(polys2, True, delta, jointype, EndType.Butt, limit) 
    else:    
        return _OffsetInternal(polys2, False, delta, jointype, endtype, limit) 

def _DistanceSqrd(pt1, pt2):
    dx = (pt1.x - pt2.x)
    dy = (pt1.y - pt2.y)
    return (dx*dx + dy*dy)

def _ClosestPointOnLine(pt, linePt1, linePt2):
    dx = linePt2.x - linePt1.x
    dy = linePt2.y - linePt1.y
    if (dx == 0 and dy == 0): 
        return DoublePoint(linePt1.x, linePt1.y)
    q = ((pt.x-linePt1.x)*dx + (pt.Y-linePt1.Y)*dy) / (dx*dx + dy*dy)
    return DoublePoint(
      (1-q)*linePt1.X + q*linePt2.X,
      (1-q)*linePt1.Y + q*linePt2.Y)

def _SlopesNearColinear(pt1, pt2, pt3, distSqrd):
    if _DistanceSqrd(pt1, pt2) > _DistanceSqrd(pt1, pt3): return False
    cpol = _ClosestPointOnLine(pt2, pt1, pt3);
    dx = pt2.x - cpol.x
    dy = pt2.y - cpol.y
    return (dx*dx + dy*dy) < distSqrd

def _PointsAreClose(pt1, pt2, distSqrd):
    dx = pt1.x - pt2.x
    dy = pt1.y - pt2.y
    return (dx * dx) + (dy * dy) <= distSqrd

def CleanPolygon(poly, distance = 1.415):
    distSqrd = distance * distance
    highI = len(poly) -1
    while (highI > 0 and _PointsEqual(poly[highI], poly[0])): highI -= 1
    if (highI < 2): return []
    pt = poly[highI]
    result = []
    i = 0
    while True:
        while (i < highI and _PointsAreClose(pt, poly[i+1], distSqrd)): i +=2
        i2 = i
        while (i < highI and (_PointsAreClose(poly[i], poly[i+1], distSqrd) or \
                _SlopesNearColinear(pt, poly[i], poly[i+1], distSqrd))): i +=1
        if i >= highI: break
        elif i != i2: continue
        pt = poly[i]
        i +=1
        result.append(pt) 
               
    if (i <= highI): result.append(poly[i])
    j = len(result)
    if (j > 2 and _SlopesNearColinear(result[j-2], result[j-1], result[0], distSqrd)): 
        del result[j-1:]
    if len(result) < 3: return []
    else: return result
    
def CleanPolygons(polys, distance = 1.415):
    result = []
    for poly in polys:
        result.append(CleanPolygon(poly, distance = 1.415))
    return result

def SimplifyPolygon(poly, fillType):
    result = []
    c = Clipper();
    c.ForceSimple = True    
    c.AddPolygon(poly, PolyType.Subject);
    c.Execute(ClipType.Union, result, fillType, fillType)
    return result

def SimplifyPolygons(polys, fillType):
    result = []
    c = Clipper();
    c.ForceSimple = True    
    c.AddPolygons(polys, PolyType.Subject);
    c.Execute(ClipType.Union, result, fillType, fillType)
    return result
