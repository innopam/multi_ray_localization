# from shapely.geometry import LineString

# # 선분 1
# line1 = LineString([(0, 0, 0), (1, 1, 1)])

# # 선분 2
# line2 = LineString([(0, 1, 0), (1, 0, 1)])

# # 선분 3
# line3 = LineString([(1, 0, 0), (0, 1, 1)])

# # 선분 4
# line4 = LineString([(0, 0, 1), (1, 1, 0)])

# # 선분 5
# line5 = LineString([(1, 0, 1), (0, 1, 0)])

# # 교차점 찾기
# intersection = line1.intersection(line2).intersection(line3).intersection(line4).intersection(line5)

# print(intersection)


import plotly.graph_objs as go
from shapely.geometry import LineString

# 선분 1
line1 = LineString([(0, 0, 0), (1, 1, 1)])

# 선분 2
line2 = LineString([(0, 1, 0), (1, 0, 1)])

# 선분 3
line3 = LineString([(1, 0, 0), (0, 1, 1)])

# 선분 4
line4 = LineString([(0, 0, 1), (1, 1, 0)])

# 선분 5
line5 = LineString([(1, 0, 1), (0, 1, 0)])

# 교차점 찾기
intersection = line1.intersection(line2).intersection(line3).intersection(line4).intersection(line5)

# 그래프 설정
fig = go.Figure()

# 선분 그리기
fig.add_trace(go.Scatter3d(
    x=[p[0] for p in line1.coords],
    y=[p[1] for p in line1.coords],
    z=[p[2] for p in line1.coords],
    mode='lines',
    name='line1',
    line=dict(color='blue', width=3)
))

fig.add_trace(go.Scatter3d(
    x=[p[0] for p in line2.coords],
    y=[p[1] for p in line2.coords],
    z=[p[2] for p in line2.coords],
    mode='lines',
    name='line2',
    line=dict(color='red', width=3)
))

fig.add_trace(go.Scatter3d(
    x=[p[0] for p in line3.coords],
    y=[p[1] for p in line3.coords],
    z=[p[2] for p in line3.coords],
    mode='lines',
    name='line3',
    line=dict(color='green', width=3)
))

fig.add_trace(go.Scatter3d(
    x=[p[0] for p in line4.coords],
    y=[p[1] for p in line4.coords],
    z=[p[2] for p in line4.coords],
    mode='lines',
    name='line4',
    line=dict(color='orange', width=3)
))

fig.add_trace(go.Scatter3d(
    x=[p[0] for p in line5.coords],
    y=[p[1] for p in line5.coords],
    z=[p[2] for p in line5.coords],
    mode='lines',
    name='line5',
    line=dict(color='purple', width=3)
))

# 교차점 시각화
if intersection.is_empty:
    print("교차점 없음")
else:
    fig.add_trace(go.Scatter3d(
        x=[intersection.x],
        y=[intersection.y],
        z=[intersection.z],
        mode='markers',
        name='intersection',
        marker=dict(color='black', size=10, symbol='circle')
    ))

# 그래프 출력
fig.show()
