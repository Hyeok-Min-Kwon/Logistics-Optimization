import pandas as pd
import numpy as np
import random
import math

# 데이터 로드
orders = pd.read_csv('data/order.csv', encoding="euc-kr")
boxes = pd.read_csv('data/box.csv', encoding="euc-kr")
locations = pd.read_csv('data/location.csv', encoding="utf-8")

# 로케이션 - 위치 --> 딕셔너리
location_dict = locations.set_index('로케이션')['위치'].to_dict()
location_dict['검수대'] = 0

# 박스 크기 딕셔너리 생성
box_size_dict = boxes.set_index('박스코드')['길이'].to_dict()

# 주문 데이터에 위치 정보 추가
orders['거리'] = orders['로케이션'].map(location_dict)

# 주문번호별 그룹핑
order_groups = orders.groupby('주문번호').indices
order_list = list(order_groups.keys())

# 카트 수 조정할 수 있게
num_cart = 6  

# 이동 거리 계산
def calculate_distance(order_sequence):
    total_distance = 0
    current_location = ['검수대'] * num_cart  # 각 카트의 초기 위치를 '검수대'로 설정
    cart_levels = [[0, 0, 0] for _ in range(num_cart)]  # 각 카트의 층별 용량을 0으로 초기화
    max_cart_size = 50  # 각 층의 최대 용량
    optimal_paths = [[] for _ in range(num_cart)]  # 각 카트의 경로를 저장할 리스트
    
    for order in order_sequence:
        order_data = orders[orders['주문번호'] == order]
        
        for cart_index in range(num_cart):
            cart_fits = True  # 현재 주문이 카트에 적합한지 여부를 나타내는 플래그
            cart_levels_copy = cart_levels[cart_index][:]  # 카트의 용량을 복사
            
            for _, row in order_data.iterrows():
                box_size = box_size_dict[row['박스코드']]
                if all(level + box_size > max_cart_size for level in cart_levels_copy):
                    cart_fits = False  # 모든 층이 용량 초과인 경우
                    break
                for level_index in range(len(cart_levels_copy)):
                    if cart_levels_copy[level_index] + box_size <= max_cart_size:
                        cart_levels_copy[level_index] += box_size
                        break
            
            if cart_fits:
                cart_levels[cart_index] = cart_levels_copy  # 복사본을 실제 카트의 용량으로 업데이트
                for _, row in order_data.iterrows():
                    total_distance += abs(location_dict[current_location[cart_index]] - location_dict[row['로케이션']])
                    current_location[cart_index] = row['로케이션']
                    optimal_paths[cart_index].append(f"주문번호 {order}")
                break
        else:
            # 현재 주문이 모든 카트에 적합하지 않은 경우 ---> 용량 초과할 떄
            min_cart_index = np.argmin([sum(levels) for levels in cart_levels])
            total_distance += abs(location_dict[current_location[min_cart_index]] - location_dict['검수대'])
            optimal_paths[min_cart_index].append("검수대")
            current_location[min_cart_index] = '검수대'
            cart_levels[min_cart_index] = [0, 0, 0]  # 용량을 초기화
            
            for _, row in order_data.iterrows():
                box_size = box_size_dict[row['박스코드']]
                for level_index in range(len(cart_levels[min_cart_index])):
                    if cart_levels[min_cart_index][level_index] + box_size <= max_cart_size:
                        cart_levels[min_cart_index][level_index] += box_size
                        break
                total_distance += abs(location_dict[current_location[min_cart_index]] - location_dict[row['로케이션']])
                current_location[min_cart_index] = row['로케이션']
                optimal_paths[min_cart_index].append(f"주문번호 {order}")
    
    # 모든 주문을 처리한 후, 각 카트를 검수대로 돌려보냄
    for cart_index in range(num_cart):
        total_distance += abs(location_dict[current_location[cart_index]] - location_dict['검수대'])
        optimal_paths[cart_index].append("검수대")
    
    return total_distance, optimal_paths

# 인접 해 생성
def generate_neighbor(order_sequence):
    neighbor = order_sequence[:]
    i, j = random.sample(range(len(neighbor)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

# 시뮬레이티드 어닐링 알고리즘
def simulated_annealing(order_list, initial_temp, cooling_rate, max_iterations):
    # 초기 해 설정
    current_solution = order_list[:]
    current_distance, current_paths = calculate_distance(current_solution)
    best_solution = current_solution[:]
    best_distance = current_distance
    best_paths = current_paths
    
    temp = initial_temp
    
    for iteration in range(max_iterations):
        neighbor = generate_neighbor(current_solution)
        neighbor_distance, neighbor_paths = calculate_distance(neighbor)
        
        if neighbor_distance < current_distance or random.uniform(0, 1) < math.exp((current_distance - neighbor_distance) / temp):
            current_solution = neighbor[:]
            current_distance = neighbor_distance
            current_paths = neighbor_paths
            
            if current_distance < best_distance:
                best_solution = current_solution[:]
                best_distance = current_distance
                best_paths = current_paths
        
        temp *= cooling_rate
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, Best Distance: {best_distance:.2f}")
    
    return best_solution, best_distance, best_paths

# 초기 설정
initial_temp = 1000
cooling_rate = 0.995
max_iterations = 10000

# 시뮬레이티드 어닐링 실행
best_order, best_distance, best_paths = simulated_annealing(order_list, initial_temp, cooling_rate, max_iterations)

# 결과 출력 및 엑셀 저장
with pd.ExcelWriter("SA_paths.xlsx") as writer:
    for cart_index in range(num_cart):  
        print(f"카트 {cart_index + 1} 경로:")
        print(" -> ".join(best_paths[cart_index]))
        
        df = pd.DataFrame({"경로": best_paths[cart_index]})
        df.to_excel(writer, sheet_name=f"카트 {cart_index + 1}", index=False)

print(f"총 이동 거리: {best_distance}")
