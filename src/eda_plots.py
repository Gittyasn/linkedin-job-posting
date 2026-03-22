import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_general_insights(df, top_n=20):
    """
    Generates INTERACTIVE Plotly charts with DARK THEME.
    """
    plots = {}
    
    # Common layout settings for Dark Theme
    layout_args = {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': dict(color='#E0E0E0')
    }

    # 1. Top Job Titles
    top_titles = df['clean_title'].value_counts().head(top_n).reset_index()
    top_titles.columns = ['Job Title', 'Count']
    
    fig_titles = px.bar(
        top_titles, 
        x='Count', 
        y='Job Title', 
        orientation='h',
        title=f"Top {top_n} Job Titles",
        text='Count',
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig_titles.update_layout(yaxis=dict(autorange="reversed"), **layout_args)
    plots['titles'] = fig_titles

    # 2. Top Locations
    if 'location' in df.columns:
        top_locs = df['location'].value_counts().head(10).reset_index()
        top_locs.columns = ['Location', 'Count']
        
        fig_locs = px.bar(
            top_locs, 
            x='Location', 
            y='Count', 
            title="Top 10 Hiring Locations",
            color='Count',
            color_continuous_scale='Magma'
        )
        fig_locs.update_layout(**layout_args)
        plots['locations'] = fig_locs

    # 3. Work Type Distribution
    if 'formatted_work_type' in df.columns:
        work_counts = df['formatted_work_type'].value_counts().reset_index()
        work_counts.columns = ['Work Type', 'Count']
        
        fig_work = px.pie(
            work_counts, 
            values='Count', 
            names='Work Type', 
            title="Work Type Distribution",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_work.update_layout(**layout_args)
        plots['work_type'] = fig_work

    return plots

def plot_salary_distribution(df):
    """
    Generates Salary distribution plot with DARK THEME.
    """
    if 'normalized_salary' not in df.columns:
        return None
        
    layout_args = {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': dict(color='#E0E0E0')
    }
        
    fig = px.histogram(
        df, 
        x="normalized_salary", 
        nbins=50, 
        title="Salary Distribution Curve",
        marginal="box", 
        color_discrete_sequence=['#00CC96']
    )
    fig.update_layout(xaxis_title="Yearly Salary ($)", yaxis_title="Count", **layout_args)
    
    return fig
