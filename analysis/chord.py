import openchord as ocd

adjacency_matrix = [[value / 100 for value in row] for row in [
  [ 0, 18,  9,  0, 23],
  [18, 0, 12,  5, 3],
  [ 9, 12,  0, 27, 10],
  [ 0,  5, 27,  0,  0],
  [23, 100, 10,  0,  0]
]]
labels = ['Canada', 'United States', 'Mexico', 'Panama', 'Brazil']

fig = ocd.Chord(adjacency_matrix, labels)
fig.show()

#fig.colormap = ['#636EFA', '#636EFA', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
fig.colormap = ['#636EFA', '#636EFA', '#AB63FA', '#AB63FA', '#00CC96']
fig.show()

fig.save_svg("figure.svg")