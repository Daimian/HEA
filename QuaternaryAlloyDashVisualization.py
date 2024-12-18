import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc

# 读取四元数据集
df = pd.read_csv('quat_4plot.csv')

# 创建 afm_lat 变量表示 afm 和 lat 的组合
def afm_lat_label(row):
    return f"{'PM' if row['afm'] == 1 else 'FM'}/{'FCC' if row['lat'] == 1 else 'BCC'}"

df['afm_lat'] = df.apply(afm_lat_label, axis=1)

# 合并元素列为列表，便于处理
element_columns = ['Element1', 'Element2', 'Element3', 'Element4']
df['Elements'] = df[element_columns].values.tolist()

# 获取所有元素的列表
all_elements = sorted(set(df[element_columns].values.flatten()))

# 定义磁性元素（根据四元数据集中的磁性元素）
magnetic_elements = ['Fe', 'Co', 'Mn', 'Cr', 'Ni']  # 可根据实际数据集调整

# 定义 afm_lat 映射和颜色比例
afm_lat_mapping = {'PM/FCC': 0, 'PM/BCC': 1, 'FM/FCC': 2, 'FM/BCC': 3}
colorscale = [
    [0.0 / 3, 'rgba(255, 99, 132, 0.7)'],   # PM/FCC
    [1.0 / 3, 'rgba(54, 162, 235, 0.7)'],   # PM/BCC
    [2.0 / 3, 'rgba(255, 206, 86, 0.7)'],   # FM/FCC
    [1.0, 'rgba(75, 192, 192, 0.7)']        # FM/BCC
]

# 初始化Dash应用，使用Bootstrap主题
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO], suppress_callback_exceptions=True)

# 函数：生成饼图
def generate_pie_chart(element):
    # 过滤包含指定磁性元素的数据
    subset = df[df['Elements'].apply(lambda x: element in x)]
    counts = subset['afm_lat'].value_counts()
    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title=f'{element} Magnetic/Crystal structure',
        color=counts.index,
        color_discrete_map={
            'PM/FCC': 'rgba(255, 99, 132, 0.7)',
            'PM/BCC': 'rgba(54, 162, 235, 0.7)',
            'FM/FCC': 'rgba(255, 206, 86, 0.7)',
            'FM/BCC': 'rgba(75, 192, 192, 0.7)'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title='afm/lat',
        title_font_size=16,
        font=dict(color='#333333')
    )
    return fig

# 定义应用布局
app.layout = dbc.Container([
    # 标题
    dbc.Row([
        dbc.Col([
            #html.H1('磁性元素组合分析（四元合金）', className='my-4')
            html.H1('Magnetic HEA quaternary database', className='my-4')
        ], width=12)
    ], justify='center'),

    # 主页面：显示磁性元素的饼图
    dbc.Row(id='main-page', className='justify-content-center', children=[
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(element, className='card-title'),
                    dcc.Graph(
                        id=f'pie-{element}',
                        figure=generate_pie_chart(element),
                        config={'displayModeBar': False}
                    ),
                    dbc.Button(
                        #'查看详情',
                        'Details',
                        id=f'button-{element}',
                        n_clicks=0,
                        className='custom-button w-100',
                        color='primary'
                    )
                ])
            ], className='chart-card')
        ], md=4, className='chart-col') for element in magnetic_elements if element in all_elements
    ]),

    # 存储选中的元素
    dcc.Store(id='selected-elements', data=[]),

    # 详情页面：初始隐藏
    dbc.Row(id='detail-page', style={'display': 'none'}, children=[
        dbc.Col([
            #html.H2('选择元素', className='my-4'),
            html.H2('Element selection', className='my-4'),
            dbc.Checklist(
                id='element-selector',
                options=[],  # 动态填充
                value=[],
                inline=True,
                className='mb-4'
            ),
            html.Div(id='visualization'),
            dbc.Button(
                '返回',
                id='back-button',
                n_clicks=0,
                className='custom-button w-100 mt-4',
                color='secondary'
            )
        ], width=12)
    ])
], fluid=True)

# 回调函数：处理“查看详情”和“返回”按钮点击事件，并设置选中的元素
@app.callback(
    Output('main-page', 'style'),
    Output('detail-page', 'style'),
    Output('selected-elements', 'data'),
    [Input('back-button', 'n_clicks')] +
    [Input(f'button-{element}', 'n_clicks') for element in magnetic_elements],
    State('selected-elements', 'data'),
    prevent_initial_call=True
)
def toggle_pages(back_clicks, *args):
    """
    处理“查看详情”和“返回”按钮点击事件，并累积固定元素。
    """
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        selected_elements = args[-1] if args[-1] else []
        if triggered_id == 'back-button':
            return {'display': 'block'}, {'display': 'none'}, []
        elif triggered_id.startswith('button-'):
            # 获取被点击的元素
            element = triggered_id.split('-')[1]
            selected_elements = [element]  # 重置为当前选中的磁性元素
            return {'display': 'none'}, {'display': 'block'}, selected_elements
        else:
            return no_update, no_update, no_update

# 回调函数：更新元素选择器选项
@app.callback(
    Output('element-selector', 'options'),
    Output('element-selector', 'value'),
    Input('detail-page', 'style'),
    State('selected-elements', 'data')
)
def update_element_selector(style, selected_elements):
    """
    在详情页面中，动态更新元素选择器的选项，排除已固定的元素。
    """
    if style and style.get('display') == 'block':
        # 选项中排除已经选中的元素
        options = [{'label': elem, 'value': elem} for elem in all_elements if elem not in selected_elements]
        return options, []
    else:
        return no_update, no_update

# 回调函数：更新可视化内容（堆叠条形图或热图）
@app.callback(
    Output('visualization', 'children'),
    Input('element-selector', 'value'),
    State('selected-elements', 'data')
)
def update_visualization(additional_selected, fixed_selected):
    """
    根据选择的元素数量，显示堆叠条形图或热图。
    """
    # 合并固定元素和额外选择的元素
    selected_elements = fixed_selected.copy()
    if additional_selected:
        selected_elements += additional_selected

    num_selected = len(selected_elements)
    total_elements = len(element_columns)

    if num_selected < 1:
        return dbc.Alert('请至少选择一个元素以生成可视化。', color='info')

    # 过滤数据：包含所有选中的元素
    filtered_df = df[df['Elements'].apply(lambda elems: all(elem in elems for elem in selected_elements))]

    if filtered_df.empty:
        return dbc.Alert('Not found.', color='warning')

    # 计算未固定的元素
    unfixed_elements = list(set(all_elements) - set(selected_elements))

    if num_selected == 1:
        # 显示堆叠条形图
        data = []
        for elem in unfixed_elements:
            elem_df = filtered_df[filtered_df['Elements'].apply(lambda x: elem in x)]
            counts = elem_df['afm_lat'].value_counts().to_dict()
            total = elem_df.shape[0]
            for afm_lat_value, count in counts.items():
                data.append({
                    'Element': elem,
                    'afm_lat': afm_lat_value,
                    'Proportion': count / total
                })
        count_df = pd.DataFrame(data)
        if count_df.empty:
            return dbc.Alert('Not found.', color='warning')
        # 创建堆叠条形图
        fig = px.bar(
            count_df,
            x='Element',
            y='Proportion',
            color='afm_lat',
            title='Distribution after fixing one element',
            # title='未固定元素对afm/lat的影响',
            color_discrete_map={
                'PM/FCC': 'rgba(255, 99, 132, 0.7)',
                'PM/BCC': 'rgba(54, 162, 235, 0.7)',
                'FM/FCC': 'rgba(255, 206, 86, 0.7)',
                'FM/BCC': 'rgba(75, 192, 192, 0.7)'
            },
            labels={'afm_lat': 'afm/lat', 'Proportion': 'proportion'},
            barmode='stack'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend_title='afm/lat',
            title_font_size=18,
            font=dict(color='#333333')
        )
        # 不再需要自定义图例，直接返回图表
        return dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': False}
                )
            ], width=12)
        ])

    elif num_selected == 2:
        # 生成热图
        if len(unfixed_elements) < 2:
            return dbc.Alert('Please select more elements.', color='warning')

        # 定义x和y轴
        x_values = unfixed_elements
        y_values = unfixed_elements

        heatmap_data = pd.DataFrame(index=y_values, columns=x_values)
        text_data = pd.DataFrame(index=y_values, columns=x_values)

        # 填充热图数据
        for x_elem in x_values:
            for y_elem in y_values:
                if x_elem == y_elem:
                    heatmap_data.loc[y_elem, x_elem] = np.nan
                    text_data.loc[y_elem, x_elem] = ''
                    continue
                subset = filtered_df[filtered_df['Elements'].apply(lambda x: x_elem in x and y_elem in x)]
                if subset.empty:
                    heatmap_data.loc[y_elem, x_elem] = np.nan
                    text_data.loc[y_elem, x_elem] = 'No data'
                else:
                    afm_lat_value = subset['afm_lat'].iloc[0]
                    B0_value = subset['B0'].iloc[0]
                    Ms_value = subset['mag'].iloc[0]
                    # Ms_value = subset['Ms'].iloc[0]
                    Tc_value = subset['TC'].iloc[0]
                    elements_combination = ', '.join(subset['Elements'].iloc[0])

                    heatmap_data.loc[y_elem, x_elem] = afm_lat_mapping.get(afm_lat_value, np.nan)
                    hover_text = f"afm/lat: {afm_lat_value}<br>B0: {B0_value}<br>Ms: {Ms_value}<br>T_c: {Tc_value}<br>Elements: {elements_combination}"
                    text_data.loc[y_elem, x_elem] = hover_text

        z_values = heatmap_data.values.astype(float)

        # 创建热图
        fig = go.Figure(data=go.Heatmap(
            x=x_values,
            y=y_values,
            z=z_values,
            text=text_data.values,
            hoverinfo='text',
            colorscale=colorscale,
            showscale=True,  # 显示颜色条
            zmin=0,
            zmax=3,
            colorbar=dict(
                title='afm/lat',
                titleside='right',
                tickmode='array',
                tickvals=[0, 1, 2, 3],
                ticktext=['PM/FCC', 'PM/BCC', 'FM/FCC', 'FM/BCC'],
                ticks='outside'
            )
        ))
        fig.update_layout(
            title='afm/lat heat map',
            xaxis=dict(
                tickangle=0
                       ),
            yaxis=dict(),
            width=800,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            font=dict(color='#333333')
        )

        # 返回布局，将信息框放在热图右侧
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        id='heatmap-graph',
                        figure=fig,
                        config={'displayModeBar': False}
                    )
                ], xs=12, sm=12, md=8, lg=8, xl=8),
                dbc.Col([
                    html.Div(id='heatmap-info', className='info-box')
                ], xs=12, sm=12, md=4, lg=4, xl=4)
            ], )
        ])
    else:
        return dbc.Alert('Please select proper number of elements.', color='info')

# 回调函数：处理热图点击事件，显示信息
@app.callback(
    Output('heatmap-info', 'children'),
    Input('heatmap-graph', 'clickData'),
    State('selected-elements', 'data'),
    State('element-selector', 'value')
)
def display_click_data(clickData, fixed_selected, additional_selected):
    """
    当用户点击热图中的某个点时，显示该点的详细信息。
    """
    if clickData is None:
        return ""
    else:
        point = clickData['points'][0]
        x_elem = point['x']
        y_elem = point['y']

        # 获取点击的 afm_lat 值
        z_value = point['z']
        afm_lat_value = [k for k, v in afm_lat_mapping.items() if v == z_value]
        afm_lat_value = afm_lat_value[0] if afm_lat_value else "unknown"

        # 合并固定元素和额外选择的元素
        if fixed_selected is None:
            fixed_selected = []
        if additional_selected is None:
            additional_selected = []
        selected_elements = fixed_selected + additional_selected + [x_elem, y_elem]

        # 筛选对应的数据
        subset = df[
            df['Elements'].apply(lambda elems: set(elems) == set(selected_elements)) &
            (df['afm_lat'] == afm_lat_value)
        ]

        if subset.empty:
            return dbc.Alert('Not found.', color='warning')
        else:
            row = subset.iloc[0]
            B0 = row['B0']
            Ms = row['mag']
            Tc = row['TC']
            afm = row['afm']
            lat = row['lat']
            elements_combination = ', '.join(row['Elements'])

            return dbc.Card([
                dbc.CardHeader("Properties"),
                dbc.CardBody([
                    #html.P(f"afm/lat: {afm_lat_value}"),
                    html.P(f"Elements: {elements_combination}"),
                    html.P(f"Magnetic state: {'PM' if afm == 1 else 'FM'}"),
                    html.P(f"Crystal structure: {'FCC' if lat == 1 else 'BCC'}"),
                    html.P(f"Bulk modulus: {B0} GPa"),
                    html.P(f"Magnetic moment: {Ms} "+u"\u03bcB"),
                    html.P(f"Curie temperature: {Tc} K"), 
                ])
            ], color='light')

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
