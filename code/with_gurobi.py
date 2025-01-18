import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# 데이터 로드
orders = pd.read_csv('data/order.csv', encoding="euc-kr")
boxes = pd.read_csv('data/box.csv', encoding="euc-kr")
locations = pd.read_csv('data/location.csv', encoding="utf-8")

# 로케이션 - 위치 --> 딕셔너리
location_dict = locations.set_index('로케이션')['위치'].to_dict()

# 검수대 0으로 추가
location_dict['검수대'] = 0

# 주문 데이터에 위치 정보 추가
orders['거리'] = orders['로케이션'].map(location_dict)

# 주문번호별 그룹핑
order_groups = orders.groupby('주문번호').indices

# 파라미터 설정
N = len(orders)  # order.csv의 행 수 (즉, 총 작업 수)
M = 6  # 카트 수
C = 3  # 카트 층 수
W = 50  # 층당 용량

# 모델 생성
model = gp.Model()

# 최대 계산 시간 10분으로 설정
model.setParam('TimeLimit', 600)  

# 결정 변수
x = model.addVars(N, M, vtype=GRB.BINARY, name="x")
y = model.addVars(N, M, C, vtype=GRB.BINARY, name="y")
r = model.addVars(N, M, vtype=GRB.BINARY, name="r")

# 거리 계산 
def calculate_distance(loc1, loc2):
    return abs(location_dict[loc1] - location_dict[loc2])

# 목적 함수 
objective = gp.quicksum(orders.loc[i, '거리'] * x[i, j] for i in range(N) for j in range(M))
objective += gp.quicksum(calculate_distance(orders.loc[i, '로케이션'], orders.loc[j, '로케이션']) * x[i, k] * x[j, k]
                        for i in range(N) for j in range(N) for k in range(M) if i != j)
objective += gp.quicksum(r[i, j] * location_dict['검수대'] for i in range(N) for j in range(M))
model.setObjective(objective, GRB.MINIMIZE)

########제약조건###########

# 주문 할당 제약
for i in range(N):
    model.addConstr(gp.quicksum(x[i, j] for j in range(M)) == 1)

# 동일한 주문번호 그룹은 하나의 사이클 내에서 처리하도록 하는 제약
for order_num, indices in order_groups.items():
    for j in range(M):
        for i in indices:
            model.addConstr(gp.quicksum(r[i, j] for i in indices) == 0)

# 카트 용량 제약: 카트의 각 층이 W보다 작아야 함
for j in range(M):
    for k in range(C):
        model.addConstr(
            gp.quicksum(
                boxes.loc[boxes['박스코드'] == orders.loc[i, '박스코드'], '길이'].values[0] * y[i, j, k]
                for i in range(N)
            ) <= W
        )

# 카트 내 층 할당 제약
for i in range(N):
    for j in range(M):
        for k in range(C):
            model.addConstr(y[i, j, k] <= x[i, j])

# 카트가 용량이 다 차면 검수대로 돌아가야 하는 제약
for j in range(M):
    for k in range(C):
        for i in range(N):
            box_length = boxes.loc[boxes['박스코드'] == orders.loc[i, '박스코드'], '길이'].values[0]
            model.addConstr(
                gp.quicksum(
                    box_length * y[i, j, k] 
                    for i in range(N)
                ) + box_length * (1 - r[i, j]) <= W
            )

# 최적화
model.optimize()

# 결과 
if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    total_distance = 0
    cart_paths = {j: ["검수대"] for j in range(M)}  
    result_data = []

    for j in range(M):
        cart_distance = 0
        previous_location = '검수대'
        load = [0] * C
        for i in range(N):
            if x[i, j].x > 0:
                current_location = orders.loc[i, '로케이션']
                cart_paths[j].append(f"주문번호{orders.loc[i, '주문번호']}")
                cart_distance += calculate_distance(previous_location, current_location)
                
                # 각 층의 용량 업데이트
                for k in range(C):
                    load[k] += boxes.loc[boxes['박스코드'] == orders.loc[i, '박스코드'], '길이'].values[0]

                # 용량 초과 시 검수대로 돌아가기
                if all(l > W for l in load):
                    cart_distance += calculate_distance(current_location, '검수대')
                    cart_paths[j].append("검수대")
                    result_data.append((j+1, ' -> '.join(cart_paths[j]), cart_distance))
                    load = [0] * C  # 카트를 비우고 다시 시작
                    previous_location = '검수대'
                else:
                    previous_location = current_location

        # 마지막으로 검수대로 돌아가기
        cart_distance += calculate_distance(previous_location, '검수대')
        cart_paths[j].append("검수대")
        total_distance += cart_distance

        result_data.append((j+1, ' -> '.join(cart_paths[j]), cart_distance))

        print(f"카트 {j+1}: {' -> '.join(cart_paths[j])} (이동거리: {cart_distance})")

    print(f"총 이동거리: {total_distance}")

    # 엑셀로 저장
    result_df = pd.DataFrame(result_data, columns=['카트 번호', '경로', '이동거리'])
    result_df.to_excel('카트6대_30분.xlsx', index=False)

else:
    print("최적화 문제가 해결되지 않았습니다.")

