"""
What-If Analysis Module
Allows users to explore how changes in their profile affect salary predictions
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class WhatIfAnalyzer:
    """Performs what-if analysis on salary predictions"""
    
    def __init__(self, predict_function, base_form_data):
        """
        Initialize analyzer
        
        Args:
            predict_function: Function that takes form_data and returns prediction
            base_form_data: Current user profile data
        """
        self.predict_function = predict_function
        self.base_form_data = base_form_data.copy()
        self.base_prediction = None
    
    def set_base_prediction(self, prediction):
        """Set the baseline prediction"""
        self.base_prediction = prediction
    
    def analyze_skill_impact(self, new_skills):
        """
        Analyze impact of adding new skills
        
        Args:
            new_skills: List of skills to add
            
        Returns:
            dict with analysis results
        """
        modified_data = self.base_form_data.copy()
        current_skills = modified_data.get('skills', '').split('|')
        current_skills = [s for s in current_skills if s]
        
        # Add new skills
        all_skills = list(set(current_skills + new_skills))
        modified_data['skills'] = '|'.join(all_skills)
        modified_data['num_skills'] = len(all_skills)
        
        # Get new prediction
        new_prediction = self.predict_function(modified_data)
        
        return {
            'new_prediction': new_prediction['predicted'],
            'change': new_prediction['predicted'] - self.base_prediction,
            'percent_change': ((new_prediction['predicted'] - self.base_prediction) / self.base_prediction) * 100,
            'added_skills': new_skills
        }
    
    def analyze_experience_growth(self, years_ahead):
        """
        Project salary growth over time
        
        Args:
            years_ahead: Number of years to project
            
        Returns:
            dict with projections
        """
        projections = []
        current_exp = self.base_form_data.get('years_of_experience', 0)
        
        for year in range(years_ahead + 1):
            modified_data = self.base_form_data.copy()
            modified_data['years_of_experience'] = current_exp + year
            
            prediction = self.predict_function(modified_data)
            projections.append({
                'year': year,
                'experience': current_exp + year,
                'salary': prediction['predicted']
            })
        
        return projections
    
    def analyze_location_change(self, new_location):
        """
        Analyze impact of relocating
        
        Args:
            new_location: New location string
            
        Returns:
            dict with analysis results
        """
        modified_data = self.base_form_data.copy()
        modified_data['location'] = new_location
        
        new_prediction = self.predict_function(modified_data)
        
        return {
            'new_location': new_location,
            'new_prediction': new_prediction['predicted'],
            'change': new_prediction['predicted'] - self.base_prediction,
            'percent_change': ((new_prediction['predicted'] - self.base_prediction) / self.base_prediction) * 100
        }
    
    def analyze_promotion(self, new_seniority):
        """
        Analyze impact of promotion
        
        Args:
            new_seniority: New seniority level
            
        Returns:
            dict with analysis results
        """
        modified_data = self.base_form_data.copy()
        modified_data['seniority_level'] = new_seniority
        
        new_prediction = self.predict_function(modified_data)
        
        return {
            'new_seniority': new_seniority,
            'new_prediction': new_prediction['predicted'],
            'change': new_prediction['predicted'] - self.base_prediction,
            'percent_change': ((new_prediction['predicted'] - self.base_prediction) / self.base_prediction) * 100
        }
    
    def analyze_education_upgrade(self, new_education):
        """
        Analyze impact of additional education
        
        Args:
            new_education: New education level
            
        Returns:
            dict with analysis results
        """
        modified_data = self.base_form_data.copy()
        modified_data['education_level'] = new_education
        
        new_prediction = self.predict_function(modified_data)
        
        return {
            'new_education': new_education,
            'new_prediction': new_prediction['predicted'],
            'change': new_prediction['predicted'] - self.base_prediction,
            'percent_change': ((new_prediction['predicted'] - self.base_prediction) / self.base_prediction) * 100
        }


def display_whatif_analysis(analyzer, artifacts, all_skills, locations, seniority_levels, education_levels, key_prefix=""):
    """
    Display interactive what-if analysis in Streamlit
    
    Args:
        analyzer: WhatIfAnalyzer instance
        artifacts: Model artifacts
        all_skills: List of available skills
        locations: List of available locations
        seniority_levels: List of seniority levels
        education_levels: List of education levels
    """
    st.markdown("---")
    st.markdown("## 🎯 What-If Analysis")
    st.markdown("Explore how changes to your profile would impact your salary")
    
    # Create tabs for different scenarios
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Learn New Skills",
        "📈 Career Growth",
        "📍 Relocate",
        "🚀 Get Promoted",
        "🎓 More Education"
    ])
    
    # Tab 1: Skills Impact
    with tab1:
        st.markdown("### What if you learned new skills?")
        
        current_skills = analyzer.base_form_data.get('skills', '').split('|')
        current_skills = [s for s in current_skills if s]
        available_skills = [s for s in all_skills if s not in current_skills]
        
        selected_new_skills = st.multiselect(
            "Select skills to add:",
            available_skills,
            key=f"{key_prefix}whatif_new_skills",
            help="Choose skills you're considering learning"
        )
        
        if selected_new_skills:
            result = analyzer.analyze_skill_impact(selected_new_skills)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Current Salary",
                    f"${analyzer.base_prediction:,.0f}"
                )
            with col2:
                st.metric(
                    "New Salary",
                    f"${result['new_prediction']:,.0f}",
                    f"${result['change']:,.0f}"
                )
            with col3:
                st.metric(
                    "Increase",
                    f"{result['percent_change']:.1f}%",
                    f"${result['change']:,.0f}"
                )
            
            if result['change'] > 0:
                st.success(f"✅ Learning {', '.join(selected_new_skills)} could increase your salary by ${result['change']:,.0f}!")
            else:
                st.info("💡 These skills may not significantly impact your salary in your current role.")
    
    # Tab 2: Career Growth Projection
    with tab2:
        st.markdown("### Project your salary growth over time")
        
        years_ahead = st.slider(
            "Years into the future:",
            min_value=1,
            max_value=10,
            value=5,
            key=f"{key_prefix}whatif_years_ahead",
            help="See how your salary might grow with experience"
        )
        
        projections = analyzer.analyze_experience_growth(years_ahead)
        
        # Create growth chart
        df_proj = pd.DataFrame(projections)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_proj['year'],
            y=df_proj['salary'],
            mode='lines+markers',
            name='Projected Salary',
            line=dict(color='#4f8ef7', width=3),
            marker=dict(size=10),
            text=[f"${s:,.0f}" for s in df_proj['salary']],
            textposition='top center'
        ))
        
        fig.update_layout(
            title=f"Salary Growth Projection ({years_ahead} Years)",
            xaxis_title="Years from Now",
            yaxis_title="Annual Salary ($)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf0', size=11),
            height=400,
            xaxis=dict(gridcolor='#2a3040'),
            yaxis=dict(gridcolor='#2a3040')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        final_salary = projections[-1]['salary']
        total_growth = final_salary - analyzer.base_prediction
        avg_annual_growth = total_growth / years_ahead
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Salary", f"${analyzer.base_prediction:,.0f}")
        with col2:
            st.metric(f"Salary in {years_ahead} Years", f"${final_salary:,.0f}")
        with col3:
            st.metric("Avg Annual Growth", f"${avg_annual_growth:,.0f}")
    
    # Tab 3: Location Change
    with tab3:
        st.markdown("### What if you relocated?")
        
        current_location = analyzer.base_form_data.get('location', 'San Francisco, CA')
        
        new_location = st.selectbox(
            "Select new location:",
            [loc for loc in locations if loc != current_location],
            key=f"{key_prefix}whatif_new_location",
            help="See how moving would affect your salary"
        )
        
        if st.button("Analyze Relocation", key=f"{key_prefix}analyze_location"):
            result = analyzer.analyze_location_change(new_location)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="background: rgba(79, 142, 247, 0.1); border-radius: 10px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 0.85rem; color: #7a829a;">Current Location</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #e8eaf0; margin: 0.5rem 0;">{current_location}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #4f8ef7;">${analyzer.base_prediction:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                color = "#34d399" if result['change'] > 0 else "#f87171"
                arrow = "↑" if result['change'] > 0 else "↓"
                st.markdown(f"""
                <div style="background: rgba(79, 142, 247, 0.1); border-radius: 10px; padding: 1.5rem; text-align: center;">
                    <div style="font-size: 0.85rem; color: #7a829a;">New Location</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #e8eaf0; margin: 0.5rem 0;">{new_location}</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {color};">${result['new_prediction']:,.0f}</div>
                    <div style="font-size: 0.9rem; color: {color}; margin-top: 0.5rem;">
                        {arrow} ${abs(result['change']):,.0f} ({result['percent_change']:+.1f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            if result['change'] > 0:
                st.success(f"✅ Moving to {new_location} could increase your salary by ${result['change']:,.0f}!")
            elif result['change'] < 0:
                st.warning(f"⚠️ Moving to {new_location} might decrease your salary by ${abs(result['change']):,.0f}")
            else:
                st.info("💡 This location change would have minimal impact on your salary")
    
    # Tab 4: Promotion
    with tab4:
        st.markdown("### What if you got promoted?")
        
        current_seniority = analyzer.base_form_data.get('seniority_level', 'Mid-level')
        current_idx = seniority_levels.index(current_seniority) if current_seniority in seniority_levels else 2
        
        # Show only higher seniority levels
        higher_levels = seniority_levels[current_idx + 1:] if current_idx < len(seniority_levels) - 1 else []
        
        if higher_levels:
            new_seniority = st.selectbox(
                "Select promotion level:",
                higher_levels,
                key=f"{key_prefix}whatif_new_seniority",
                help="See how a promotion would affect your salary"
            )
            
            if st.button("Analyze Promotion", key=f"{key_prefix}analyze_promotion"):
                result = analyzer.analyze_promotion(new_seniority)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current Level",
                        current_seniority,
                        f"${analyzer.base_prediction:,.0f}"
                    )
                with col2:
                    st.metric(
                        "New Level",
                        new_seniority,
                        f"${result['new_prediction']:,.0f}"
                    )
                with col3:
                    st.metric(
                        "Salary Increase",
                        f"{result['percent_change']:.1f}%",
                        f"${result['change']:,.0f}"
                    )
                
                st.success(f"🚀 Promotion to {new_seniority} could increase your salary by ${result['change']:,.0f}!")
        else:
            st.info("💡 You're already at the highest seniority level!")
    
    # Tab 5: Education
    with tab5:
        st.markdown("### What if you pursued higher education?")
        
        current_education = analyzer.base_form_data.get('education_level', 'Bachelor')
        current_idx = education_levels.index(current_education) if current_education in education_levels else 2
        
        # Show only higher education levels
        higher_education = education_levels[current_idx + 1:] if current_idx < len(education_levels) - 1 else []
        
        if higher_education:
            new_education = st.selectbox(
                "Select education level:",
                higher_education,
                key=f"{key_prefix}whatif_new_education",
                help="See how additional education would affect your salary"
            )
            
            if st.button("Analyze Education Impact", key=f"{key_prefix}analyze_education"):
                result = analyzer.analyze_education_upgrade(new_education)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="background: rgba(79, 142, 247, 0.1); border-radius: 10px; padding: 1.5rem; text-align: center;">
                        <div style="font-size: 0.85rem; color: #7a829a;">Current Education</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #e8eaf0; margin: 0.5rem 0;">{current_education}</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #4f8ef7;">${analyzer.base_prediction:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: rgba(52, 211, 153, 0.1); border-radius: 10px; padding: 1.5rem; text-align: center;">
                        <div style="font-size: 0.85rem; color: #7a829a;">With {new_education}</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #e8eaf0; margin: 0.5rem 0;">{new_education}</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #34d399;">${result['new_prediction']:,.0f}</div>
                        <div style="font-size: 0.9rem; color: #34d399; margin-top: 0.5rem;">
                            ↑ ${result['change']:,.0f} ({result['percent_change']:+.1f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ROI calculation (rough estimate)
                education_costs = {
                    'Master': 50000,
                    'MBA': 100000,
                    'PhD': 80000
                }
                
                cost = education_costs.get(new_education, 50000)
                years_to_roi = cost / result['change'] if result['change'] > 0 else float('inf')
                
                st.markdown("### 💰 Return on Investment")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Cost", f"${cost:,.0f}")
                with col2:
                    st.metric("Annual Salary Increase", f"${result['change']:,.0f}")
                with col3:
                    if years_to_roi < 10:
                        st.metric("Years to ROI", f"{years_to_roi:.1f}")
                    else:
                        st.metric("Years to ROI", "10+")
                
                if result['change'] > 0:
                    st.success(f"🎓 Pursuing a {new_education} could increase your salary by ${result['change']:,.0f} annually!")
                else:
                    st.info("💡 Additional education may not significantly impact your salary in your current role.")
        else:
            st.info("💡 You're already at the highest education level!")
