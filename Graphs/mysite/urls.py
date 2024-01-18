from django.urls import path,include
from .views import *

urlpatterns = [
    path('', plotly_all_charts, name='index'),
    path('plot/', plotly_gamma_plot, name='plot'),
    path('vega_plot/',plotly_vega_chart,name="vega_plot"),
    path('delta_plot/',plotly_delta_chart,name="delta_plot"),
    path('prices_plot/',plotly_prices_chart,name="price_plot"),
    path('ce_gxoi_plot/',plotly_CE_gxoi_chart,name="ce_gxoi_plot"),
    path('pe_gxoi_plot/',plotly_pe_gxoi_chart,name="pe_gxoi_plot"),
    path('ce_vxoi_plot/',plotly_ce_vxoi_chart,name="ce_vxoi_plot"),
    path('pe_vxoi_plot/',plotly_pe_vxoi_chart,name="pe_vxoi_plot"),
    # plotly_gxoi_chart
    # plotly_all_charts

]