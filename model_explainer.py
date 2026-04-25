"""
Model Explainability Module using SHAP
Provides interpretable explanations for salary predictions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")


class SalaryExplainer:
    """Explains salary predictions using SHAP values"""
    
    def __init__(self, model, feature_names, background_data=None):
        """
        Initialize explainer
        
        Args:
            model: Trained model (should have predict method)
            feature_names: List of feature names
            background_data: Sample of training data for SHAP (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        if SHAP_AVAILABLE:
            try:
                # Check if it's a stacking ensemble
                if hasattr(model, 'estimators_') and hasattr(model, 'final_estimator_'):
                    # For stacking models, use the first tree-based estimator
                    print("Detected stacking ensemble, using first tree-based estimator for SHAP")
                    tree_model = None
                    for est in model.estimators_:
                        # Check if it's a tree-based model
                        if hasattr(est, 'feature_importances_'):
                            tree_model = est
                            break
                    
                    if tree_model is not None:
                        self.explainer = shap.TreeExplainer(tree_model)
                        print(f"Using {tree_model.__class__.__name__} for SHAP explanations")
                    else:
                        raise Exception("No tree-based model found in stacking ensemble")
                else:
                    # Use TreeExplainer for tree-based models (faster, doesn't need background data)
                    self.explainer = shap.TreeExplainer(model)
            except Exception as e:
                print(f"TreeExplainer failed: {e}")
                # Fallback to KernelExplainer (slower but works for any model)
                # This requires background_data
                if background_data is not None:
                    try:
                        print("Falling back to KernelExplainer...")
                        self.explainer = shap.KernelExplainer(
                            model.predict, 
                            shap.sample(background_data, 100)
                        )
                    except Exception as e2:
                        print(f"Could not initialize SHAP explainer: {e2}")
                else:
                    print("Warning: TreeExplainer failed and no background_data provided for KernelExplainer")
    
    def explain_prediction(self, X, base_value=None):
        """
        Get SHAP values for a prediction
        
        Args:
            X: Feature vector (numpy array or pandas DataFrame)
            base_value: Base prediction value (optional)
            
        Returns:
            dict with explanation data
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(X)
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Get base value
            if base_value is None:
                try:
                    base_value = self.explainer.expected_value
                    if isinstance(base_value, np.ndarray):
                        base_value = base_value[0]
                except:
                    base_value = 0
            
            # Create feature importance ranking
            feature_impacts = []
            for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values)):
                if isinstance(X, pd.DataFrame):
                    feature_value = X.iloc[0, i]
                else:
                    feature_value = X[0, i] if len(X.shape) > 1 else X[i]
                
                feature_impacts.append({
                    'feature': feature,
                    'value': feature_value,
                    'shap_value': shap_val,
                    'impact': abs(shap_val)
                })
            
            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: x['impact'], reverse=True)
            
            return {
                'shap_values': shap_values,
                'base_value': base_value,
                'feature_impacts': feature_impacts,
                'top_features': feature_impacts[:10],
                'available': True
            }
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(X)
    
    def _fallback_explanation(self, X):
        """Fallback when SHAP is not available"""
        return {
            'available': False,
            'message': 'SHAP explanations not available. Install shap package for detailed explanations.'
        }
    
    def plot_waterfall(self, explanation, predicted_value):
        """
        Create waterfall chart showing feature contributions
        
        Args:
            explanation: Output from explain_prediction
            predicted_value: Final predicted salary
            
        Returns:
            Plotly figure
        """
        if not explanation.get('available', False):
            return None
        
        top_features = explanation['top_features'][:10]
        base_value = explanation['base_value']
        
        # Prepare data for waterfall
        features = ['Base Value']
        values = [base_value]
        
        cumulative = base_value
        for feat in top_features:
            features.append(self._format_feature_name(feat['feature']))
            values.append(feat['shap_value'])
            cumulative += feat['shap_value']
        
        features.append('Final Prediction')
        values.append(predicted_value - cumulative)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Salary Impact",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(top_features) + ["total"],
            x=features,
            textposition="outside",
            text=[f"${v:,.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#34d399"}},
            decreasing={"marker": {"color": "#f87171"}},
            totals={"marker": {"color": "#4f8ef7"}}
        ))
        
        fig.update_layout(
            title="How Features Impact Your Salary Prediction",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf0', size=11),
            height=500,
            xaxis=dict(tickangle=-45),
            yaxis=dict(title="Salary Impact ($)", gridcolor='#2a3040')
        )
        
        return fig
    
    def plot_feature_importance(self, explanation):
        """
        Create horizontal bar chart of top feature impacts
        
        Args:
            explanation: Output from explain_prediction
            
        Returns:
            Plotly figure
        """
        if not explanation.get('available', False):
            return None
        
        top_features = explanation['top_features'][:15]
        
        features = [self._format_feature_name(f['feature']) for f in top_features]
        impacts = [f['shap_value'] for f in top_features]
        colors = ['#34d399' if i > 0 else '#f87171' for i in impacts]
        
        fig = go.Figure(go.Bar(
            y=features[::-1],
            x=impacts[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"${abs(i):,.0f}" for i in impacts[::-1]],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 15 Features Impacting Your Salary",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e8eaf0', size=11),
            height=500,
            xaxis=dict(title="Impact on Salary ($)", gridcolor='#2a3040', zeroline=True, zerolinecolor='#7a829a'),
            yaxis=dict(title="")
        )
        
        return fig
    
    def get_feature_insights(self, explanation):
        """
        Generate human-readable insights from SHAP values
        
        Args:
            explanation: Output from explain_prediction
            
        Returns:
            List of insight strings
        """
        if not explanation.get('available', False):
            return []
        
        insights = []
        top_features = explanation['top_features'][:5]
        
        for feat in top_features:
            feature_name = self._format_feature_name(feat['feature'])
            impact = feat['shap_value']
            
            if impact > 0:
                insights.append(f"✅ **{feature_name}** increases your salary by **${abs(impact):,.0f}**")
            else:
                insights.append(f"⚠️ **{feature_name}** decreases your salary by **${abs(impact):,.0f}**")
        
        return insights
    
    def _format_feature_name(self, feature):
        """Format feature name for display"""
        # Remove prefixes
        feature = feature.replace('skill_', '').replace('_', ' ')
        
        # Capitalize
        feature = ' '.join(word.capitalize() for word in feature.split())
        
        # Special cases
        replacements = {
            'Exp': 'Experience',
            'Edu': 'Education',
            'Num': 'Number of',
            'Gpa': 'GPA',
            'Seniority X Exp': 'Seniority × Experience',
            'Skills X Exp': 'Skills × Experience',
            'Target Enc': 'Average Salary',
            'Freq': 'Frequency'
        }
        
        for old, new in replacements.items():
            feature = feature.replace(old, new)
        
        return feature


def display_shap_explanation(explainer, X, predicted_salary, feature_names):
    """
    Display SHAP explanation in Streamlit
    
    Args:
        explainer: SalaryExplainer instance
        X: Feature vector
        predicted_salary: Predicted salary value
        feature_names: List of feature names
    """
    if not SHAP_AVAILABLE:
        st.info("💡 Install SHAP for detailed prediction explanations: `pip install shap`")
        return
    
    with st.spinner("Generating explanation..."):
        explanation = explainer.explain_prediction(X)
    
    if not explanation.get('available', False):
        st.warning(explanation.get('message', 'Explanations not available'))
        return
    
    st.markdown("---")
    st.markdown("## 🔍 Why This Salary?")
    st.markdown("Understanding what drives your salary prediction using AI explainability (SHAP)")
    
    # Key insights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💡 Key Insights")
        insights = explainer.get_feature_insights(explanation)
        for insight in insights:
            st.markdown(insight)
    
    with col2:
        st.markdown("### 📊 Base Salary")
        base_val = explanation['base_value']
        st.markdown(f"""
        <div style="background: rgba(79, 142, 247, 0.1); border-radius: 10px; padding: 1rem; text-align: center;">
            <div style="font-size: 0.85rem; color: #7a829a;">Average Salary</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #4f8ef7;">${base_val:,.0f}</div>
            <div style="font-size: 0.75rem; color: #7a829a; margin-top: 0.5rem;">
                Your features adjust this base value
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Waterfall chart
    st.markdown("### 📈 Feature Impact Breakdown")
    waterfall_fig = explainer.plot_waterfall(explanation, predicted_salary)
    if waterfall_fig:
        st.plotly_chart(waterfall_fig, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🎯 Top Contributing Features")
    importance_fig = explainer.plot_feature_importance(explanation)
    if importance_fig:
        st.plotly_chart(importance_fig, use_container_width=True)
    
    # Detailed breakdown
    with st.expander("📋 Detailed Feature Breakdown"):
        top_features = explanation['top_features'][:20]
        
        df_features = pd.DataFrame([
            {
                'Feature': explainer._format_feature_name(f['feature']),
                'Your Value': f"{f['value']:.2f}" if isinstance(f['value'], (int, float)) else str(f['value']),
                'Impact': f"${f['shap_value']:,.0f}",
                'Direction': '↑ Increases' if f['shap_value'] > 0 else '↓ Decreases'
            }
            for f in top_features
        ])
        
        st.dataframe(df_features, use_container_width=True, hide_index=True)
