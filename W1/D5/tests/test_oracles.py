from fractions import Fraction
from itertools import product, combinations
from day5.tasks import TASKS

def test_task_oracles_independently_verified():
    expected={t['id']:t['expected'] for t in TASKS}
    knights=[]
    for A,B,C,D in product([False,True],repeat=4):
        if [A,B,C,D]==[not B,C==D,not A and not B,(A+B)==1]: knights.append((A,B,C,D))
    assert knights==[(True,False,False,True)] and expected['T1']=='A,D'
    strings=sum(sum(bits)==5 and all(not(bits[i] and bits[i+1]) for i in range(11)) for bits in product((0,1),repeat=12))
    assert strings==int(expected['T2'])==56
    assert sum(sum(n%d==0 for d in (6,10,15))==1 for n in range(1,1001))==int(expected['T3'])
    items=[(3,4),(4,5),(5,7),(8,12),(9,13)]
    best=max(sum(v for w,v in combo) for k in range(6) for combo in combinations(items,k) if sum(w for w,v in combo)<=15)
    assert best==int(expected['T4'])==21
    assert Fraction(1,100)*Fraction(9,10)/(Fraction(1,100)*Fraction(9,10)+Fraction(99,100)*Fraction(5,100))==Fraction(expected['T5'])
    a=[3,4,-1,0,6,2,3,7,5,8,9,1]
    best=max(len(seq) for n in range(len(a)+1) for seq in combinations(a,n) if all(x<y for x,y in zip(seq,seq[1:])))
    assert best==int(expected['T6'])==7
    assert [Fraction(81,87)>Fraction(234,270),Fraction(192,263)>Fraction(55,80),Fraction(273,350)>Fraction(289,350)]==[True,True,False]
    assert expected['T7']=='A,A,B' and expected['T8']=='14'

def test_schedule_oracle_is_minimal_not_just_a_feasible_answer():
    durations=[3,5,4,6,4,3]; deps=[[],[],[0],[0],[1,2],[3,4]]
    def feasible(horizon):
        starts=[]; occupied=[0]*horizon
        def search(i):
            if i==6:return True
            earliest=max((starts[j]+durations[j] for j in deps[i]),default=0)
            for start in range(earliest,horizon-durations[i]+1):
                interval=range(start,start+durations[i])
                if any(occupied[t]>=2 for t in interval):continue
                starts.append(start)
                for t in interval:occupied[t]+=1
                if search(i+1):return True
                for t in interval:occupied[t]-=1
                starts.pop()
            return False
        return search(0)
    assert not feasible(13) and feasible(14)
    assert TASKS[-1]['expected']=='14'
