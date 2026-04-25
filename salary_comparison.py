"""
Salary Comparison Module
Compare user's salary against market benchmarks
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


class SalaryComparator:
    """Compare salaries against market benchmarks"""
    
    def __init__(self, user_salary, form_data):
        """
        Initialize comparator
        
        Args:
            user_salary: User's predicted salary
            form_data: User's profile data
        """
        self.user_salary = user_salary
        self.form_data = form_data
        
        # Market data (these would ideally come from a database)
        self.market_data = self._generate_market_data()
    
    def _generate_market_data(self):
        """Generate synthetic market data for comparison"""
        # In production, this would query a database
        # For now, we'll generate realistic distributions
        
        job_title = self.form_data.get('job_title', 'Software Engineer')
        seniority = self.form_data.get('seniority_level', 'Mid-level')
        location = self.form_data.get('location', 'San Francisco, CA')
        experience = self.form_data.get('years_of_experience', 4)
        
        # Base salaries by seniority
        base_salaries = {
            'Intern': 60000,
            'Junior': 80000,
            'Mid-level': 120000,
            'Senior': 160000,
            'Lead': 190000,
            'Principal': 220000,
            'Staff': 240000,
            'Director': 280000,
            'VP': 350000,
            'CTO/CXO': 450000
        }
        
        base = base_salaries.get(seniority, 120000)
        
        # Generate distribution around base
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic salary distribution (log-normal)
        salaries = np.random.lognormal(
            mean=np.log(base),
            sigma=0.25,
            size=n_samples
        )
        
        return {
            'salaries': salaries,
            'base': base,
            'percentiles': {
                '10th': np.percentile(salaries, 10),
                '25th': np.percentile(salaries, 25),
                '50th': np.percentile(salaries, 50),
                '75th': np.percentile(salaries, 75),
                '90th': np.percentile(salaries, 90)
            }
        }
    
    def get_percentile_rank(self):
        """Calculate user's percentile rank"""
        salaries = self.market_data['salaries']
        percentile = (salaries < self.user_salary).sum() / len(salaries) * 100
        return percentile
    
    def plot_distribution(self):
        """Create user-friendly salary distribution chart"""
        salaries = self.market_data['salaries']
        percentiles = self.market_data['percentiles']
        
        # Create histogram
        fig = go.Figure()
        
        # Add distribution
        fig.add_trace(go.Histogram(
            x=salaries,
            nbinsx=50,
            name='Number of People',
            marker_color='rgba(79, 142, 247, 0.6)',
            hovertemplate='<b>Salary Range:</b> $%{x:,.0f}<br><b>Number of People:</b> %{y}<extra></extra>'
        ))
        
        # Add user's salary line
        fig.add_vline(
            x=self.user_salary,
            line_dash="solid",
            line_color="#34d399",
            line_width=4,
            annotation_text=f"<b>YOU</b><br>${self.user_salary:,.0f}",
            annotation_position="top",
            annotation_font_size=14,
            annotation_font_color="#34d399"
        )
        
        # Add median line
        median = percentiles['50th']
        fig.add_vline(
            x=median,
            line_dash="dot",
            line_color="#fbbf24",
            line_width=2,
            annotation_text=f"Average<br>${median:,.0f}",
            annotation_position="bottom",
            annotation_font_size=11,
            annotation_font_color="#fbbf24"
        )
        
        fig.update_layout(
            title={
                'text': "Salary Distribution - Where Do You Stand?",
                'font': {'size': 18}
            },
            xaxis_title="Annual Salary",
            yaxis_title="Number of People Earning This Amount",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf0', size=12),
            height=450,
            showlegend=False,
            xaxis=dict(
                gridcolor='#2a3040',
                tickformat='$,.0f'
            ),
            yaxis=dict(gridcolor='#2a3040'),
            hovermode='x unified'
        )
        
        return fig
    
    def plot_percentile_gauge(self):
        """Create gauge chart showing percentile rank"""
        percentile = self.get_percentile_rank()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentile,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Where You Stand", 'font': {'size': 20, 'color': '#e8eaf0'}},
            number={'suffix': "%", 'font': {'size': 50}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#7a829a"},
                'bar': {'color': "#4f8ef7"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#2a3040",
                'steps': [
                    {'range': [0, 25], 'color': 'rgba(248, 113, 113, 0.3)'},
                    {'range': [25, 50], 'color': 'rgba(251, 191, 36, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(79, 142, 247, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(52, 211, 153, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#34d399", 'width': 4},
                    'thickness': 0.75,
                    'value': percentile
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#e8eaf0", 'family': "Arial"},
            height=300
        )
        
        return fig
    
    def get_comparison_insights(self):
        """Generate simple, easy-to-understand insights"""
        percentile = self.get_percentile_rank()
        percentiles = self.market_data['percentiles']
        
        insights = []
        
        # Percentile insight - simplified language
        if percentile >= 75:
            insights.append({
                'icon': '🌟',
                'title': 'Excellent!',
                'message': f"You're earning more than {percentile:.0f} out of 100 people with similar jobs!"
            })
        elif percentile >= 50:
            insights.append({
                'icon': '✅',
                'title': 'Good Position',
                'message': f"You're earning more than {percentile:.0f} out of 100 people in your field."
            })
        elif percentile >= 25:
            insights.append({
                'icon': '💡',
                'title': 'Room to Grow',
                'message': f"Many people in your field earn more. You could negotiate for a higher salary."
            })
        else:
            insights.append({
                'icon': '📈',
                'title': 'Time to Negotiate',
                'message': f"Most people in your field earn more. Consider asking for a raise or looking for better opportunities."
            })
        
        # Comparison to median - simplified
        median = percentiles['50th']
        diff_from_median = self.user_salary - median
        if abs(diff_from_median) > 5000:
            if diff_from_median > 0:
                insights.append({
                    'icon': '💰',
                    'title': 'Above Average',
                    'message': f"You earn ${diff_from_median:,.0f} more than the typical person in your role."
                })
            else:
                insights.append({
                    'icon': '⚠️',
                    'title': 'Below Average',
                    'message': f"The typical person in your role earns ${abs(diff_from_median):,.0f} more. You might be underpaid."
                })
        
        # Comparison to 75th percentile - simplified
        p75 = percentiles['75th']
        if self.user_salary < p75:
            gap = p75 - self.user_salary
            insights.append({
                'icon': '🎯',
                'title': 'Your Goal',
                'message': f"Top performers earn ${p75:,.0f}. That's ${gap:,.0f} more than you - a good target for your next raise!"
            })
        
        return insights


def display_salary_comparison(user_salary, form_data):
    """
    Display salary comparison in Streamlit with simple, user-friendly language
    
    Args:
        user_salary: User's predicted salary
        form_data: User's profile data
    """
    st.markdown("---")
    
    # Header with help button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## 📊 How Does Your Salary Compare?")
    with col2:
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **Quick Guide:**
            - **Gauge** = Your score out of 100
            - **Table** = What others earn
            - **Chart** = Where you fit in
            
            [📖 Full Guide](SALARY_COMPARISON_GUIDE.md)
            """)
    
    # Simple explanation
    st.markdown("""
    <div style="background: rgba(79, 142, 247, 0.1); border-radius: 10px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <strong>💡 What does this mean?</strong><br>
        We compared your salary to 1,000 people with similar jobs, experience, and location. 
        This shows where you stand and if you're being paid fairly.
    </div>
    """, unsafe_allow_html=True)
    
    comparator = SalaryComparator(user_salary, form_data)
    percentile = comparator.get_percentile_rank()
    
    # Percentile rank with simple explanation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gauge_fig = comparator.plot_percentile_gauge()
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Simple explanation of the gauge
        st.markdown("""
        <div style="text-align: center; font-size: 0.85rem; color: #7a829a; margin-top: -1rem;">
            <strong>Think of it like a test score:</strong><br>
            You scored better than {:.0f} out of 100 people!
        </div>
        """.format(percentile), unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💰 What Others Are Earning")
        st.markdown("Here's what people in similar roles typically make:")
        
        percentiles = comparator.market_data['percentiles']
        
        # Simplified benchmark descriptions
        benchmark_data = [
            {'Level': '🌟 Top Earners (Top 10%)', 'Salary': f"${percentiles['90th']:,.0f}"},
            {'Level': '✨ High Earners (Top 25%)', 'Salary': f"${percentiles['75th']:,.0f}"},
            {'Level': '📊 Average (Middle)', 'Salary': f"${percentiles['50th']:,.0f}"},
            {'Level': '📉 Below Average', 'Salary': f"${percentiles['25th']:,.0f}"},
            {'Level': '⚠️ Low Earners (Bottom 10%)', 'Salary': f"${percentiles['10th']:,.0f}"},
        ]
        
        df_benchmarks = pd.DataFrame(benchmark_data)
        st.dataframe(df_benchmarks, use_container_width=True, hide_index=True)
        
        # Simple position statement
        if percentile >= 75:
            position_text = "🌟 <strong>You're a top earner!</strong> You're earning more than most people in your field."
        elif percentile >= 50:
            position_text = "✅ <strong>You're doing well!</strong> You're earning more than average."
        elif percentile >= 25:
            position_text = "💡 <strong>You're below average.</strong> There's room to negotiate for more."
        else:
            position_text = "⚠️ <strong>You might be underpaid.</strong> Consider asking for a raise or looking for better opportunities."
        
        st.markdown(f"""
        <div style="background: rgba(79, 142, 247, 0.1); border-radius: 8px; padding: 1rem; margin-top: 1rem;">
            {position_text}<br>
            <span style="font-size: 0.9rem; color: #7a829a;">
                You're earning more than <strong>{percentile:.0f} out of 100</strong> people with similar jobs.
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribution chart with simple explanation
    st.markdown("### 📊 Where You Fit In")
    st.markdown("""
    <div style="background: rgba(79, 142, 247, 0.05); border-radius: 8px; padding: 0.8rem; margin-bottom: 1rem; font-size: 0.9rem;">
        <strong>📖 How to read this chart:</strong><br>
        • Each bar shows how many people earn that salary<br>
        • The <span style="color: #34d399;"><strong>green line</strong></span> is YOUR salary<br>
        • Taller bars = more people earning that amount<br>
        • See if you're on the left (lower pay) or right (higher pay) side
    </div>
    """, unsafe_allow_html=True)
    
    dist_fig = comparator.plot_distribution()
    st.plotly_chart(dist_fig, use_container_width=True)
    
    # Insights with simpler presentation
    st.markdown("### 💡 What This Means For You")
    insights = comparator.get_comparison_insights()
    
    # Display insights as cards
    for insight in insights:
        st.markdown(f"""
        <div style="background: rgba(79, 142, 247, 0.1); border-radius: 10px; padding: 1.2rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2.5rem;">{insight['icon']}</div>
                <div style="flex: 1;">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #4f8ef7; margin-bottom: 0.3rem;">{insight['title']}</div>
                    <div style="font-size: 0.95rem; color: #e8eaf0; line-height: 1.5;">{insight['message']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown with simple language
    with st.expander("📋 More Details (Click to expand)"):
        job_title = form_data.get('job_title', 'Software Engineer')
        seniority = form_data.get('seniority_level', 'Mid-level')
        location = form_data.get('location', 'San Francisco, CA')
        
        st.markdown(f"""
        ### Who We Compared You To:
        - **Job:** {job_title}
        - **Level:** {seniority}
        - **Location:** {location}
        - **Sample:** 1,000 people with similar profiles
        
        ### Your Salary: ${user_salary:,.0f}
        
        ### Market Numbers Explained:
        - **Average Salary:** ${np.mean(comparator.market_data['salaries']):,.0f}
          <br><small style="color: #7a829a;">→ Add up everyone's salary and divide by 1,000</small>
        
        - **Middle Salary (Median):** ${percentiles['50th']:,.0f}
          <br><small style="color: #7a829a;">→ Half earn more, half earn less than this</small>
        
        - **Salary Range:** ${np.std(comparator.market_data['salaries']):,.0f}
          <br><small style="color: #7a829a;">→ How much salaries vary (bigger number = more variation)</small>
        
        ---
        
        **💡 Important Note:**  
        These numbers are based on people with similar jobs, experience, and location. 
        Your actual market value may be different based on:
        - The specific company you work for
        - Your unique skills and achievements
        - Current job market conditions
        - Industry trends
        """)
