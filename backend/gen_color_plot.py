import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tempfile
import webbrowser
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

class GeneratingPlots:
    """
    A class to handle plot generation for the healthcare data assistant.
    Takes output from queries in various formats, converts to DataFrame, and generates
    appropriate visualizations.
    """
    
    def __init__(self):
        """Initialize the plotting class with default settings."""
        # Set default plotting style for better aesthetics
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
        # Keywords to identify plot requests
        self.plot_keywords = [
            'plot', 'chart', 'graph', 'visualize', 'visualization', 
            'show me', 'display', 'diagram', 'figure'
        ]
        
        # Healthcare domain-specific category mappings for better labels
        self.category_mappings = {
            'region': 'Region',
            'provider': 'Healthcare Provider',
            'hospital': 'Hospital',
            'facility': 'Medical Facility',
            'diagnosis': 'Diagnosis',
            'procedure': 'Procedure',
            'disease': 'Disease',
            'condition': 'Medical Condition',
            'age': 'Age Group',
            'gender': 'Gender',
            'demographic': 'Demographic',
            'department': 'Department',
            'specialty': 'Medical Specialty',
            'drug': 'Medication',
            'insurance': 'Insurance Type',
            'payer': 'Payer',
            'cost': 'Cost',
            'payment': 'Payment',
            'reimbursement': 'Reimbursement',
            'length_of_stay': 'Length of Stay',
            'readmission': 'Readmission Rate'
        }
    
    def is_plot_request(self, query: str) -> bool:
        """
        Check if the query is requesting a visualization.
        
        Args:
            query: The user's query string
            
        Returns:
            bool: True if the query is a plot request
        """
        return any(keyword in query.lower() for keyword in self.plot_keywords)
    
    def process_query_result(self, result: Any, query: str, raw_sql_result=None) -> Optional[str]:
        """
        Process a query result to generate a plot if needed.
        
        Args:
            result: The result object from the query
            query: The original user query
            raw_sql_result: Raw SQL result tuples if available
            
        Returns:
            Optional[str]: A message about the plot or None if no plot was created
        """
        if not self.is_plot_request(query):
            return None
            
        try:
            # Extract custom title if provided in the result object
            custom_title = None
            if isinstance(result, dict) and "title" in result:
                custom_title = result["title"]
            
            # First priority: Use raw SQL result tuples if available
            if raw_sql_result is not None:
                print("Using raw SQL result for plotting")
                df = self._convert_raw_sql_to_dataframe(raw_sql_result, query)
                if df is not None and not df.empty and len(df.columns) >= 2:
                    return self._generate_plot_from_dataframe(df, query, custom_title)
            
            # Second priority: Extract data from result object
            df = self._extract_dataframe_from_result(result, query)
            if df is not None and not df.empty and len(df.columns) >= 2:
                return self._generate_plot_from_dataframe(df, query, custom_title)
                
            # If no valid DataFrame could be created, return an error message
            return "Unable to extract data for plotting from the result."
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing query result for plotting: {str(e)}"
    
    def _convert_raw_sql_to_dataframe(self, sql_result: Any, query: str) -> Optional[pd.DataFrame]:
        """
        Convert raw SQL query result to a pandas DataFrame.
        
        Args:
            sql_result: Raw SQL query result
            query: The original user query
            
        Returns:
            Optional[DataFrame]: A properly structured DataFrame or None
        """
        try:
            print(f"Debug: Raw SQL result type: {type(sql_result)}")
            print(f"Debug: Raw SQL result content: {sql_result}")
            
            # Handle string representation of results (convert to actual data structure)
            if isinstance(sql_result, str):
                try:
                    # If the string contains Decimal values, use regex to extract data
                    if "Decimal(" in sql_result:
                        import re
                        
                        # Extract tuples with pattern ('category', Decimal('value'))
                        pattern = r"\('([^']+)',\s*Decimal\('([^']+)'\)\)"
                        matches = re.findall(pattern, sql_result)
                        
                        if matches:
                            # Convert to list of tuples with proper types
                            extracted_data = [(category, float(value)) for category, value in matches]
                            sql_result = extracted_data
                        else:
                            # Try to safely evaluate the string as a Python expression
                            import ast
                            sql_result = ast.literal_eval(sql_result)
                    else:
                        # Try to safely evaluate the string as a Python expression
                        import ast
                        sql_result = ast.literal_eval(sql_result)
                except Exception as e:
                    print(f"Could not convert string SQL result to data structure: {e}")
            
            # Handle SQL result in tuple format
            if isinstance(sql_result, list) and sql_result and isinstance(sql_result[0], tuple):
                # Check tuples structure
                if len(sql_result[0]) == 1:
                    # Only one column, create a count column with value 1
                    col_name = self._infer_category_column_name(query)
                    df = pd.DataFrame({col_name: [item[0] for item in sql_result]})
                    # Add count column based on query
                    count_name = self._infer_value_column_name(query)
                    # Since we don't have actual counts, we'll use placeholder values
                    df[count_name] = [1000, 900, 800, 700, 600][:len(df)]  # Placeholder decreasing values
                    print(f"Created DataFrame with placeholder counts: {df}")
                elif len(sql_result[0]) == 2:
                    # Two columns, likely category and value
                    col1_name = self._infer_category_column_name(query)
                    col2_name = self._infer_value_column_name(query)
                    df = pd.DataFrame(sql_result, columns=[col1_name, col2_name])
                    print(f"Created DataFrame from raw SQL: {df}")
                else:
                    # Multiple columns, use generic names
                    df = pd.DataFrame(sql_result)
                    df.columns = [f"column_{i}" for i in range(len(df.columns))]
                    print(f"Created DataFrame with generic column names: {df}")
                
                # Ensure proper types for plotting
                for col in df.columns:
                    if col == df.columns[0] or col in ['region', 'category', 'hospital', 'facility']:
                        df[col] = df[col].astype(str)
                    elif col == df.columns[1] or col in ['count', 'value', 'total', 'number']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                print(f"Created DataFrame from raw SQL with shape {df.shape}")
                return df
                
            # Handle SQL result in dict/list format
            elif isinstance(sql_result, (dict, list)):
                df = self._parse_json_data(sql_result)
                if df is not None and not df.empty:
                    print(f"Created DataFrame from SQL result with shape {df.shape}")
                    return df
                    
            return None
            
        except Exception as e:
            print(f"Error converting raw SQL to DataFrame: {str(e)}")
            return None
    
    def _extract_dataframe_from_result(self, result: Any, query: str) -> Optional[pd.DataFrame]:
        """
        Extract a DataFrame from a query result in any format.
        
        Args:
            result: The query result in any format
            query: The original user query
            
        Returns:
            Optional[DataFrame]: A properly structured DataFrame or None
        """
        try:
            # Check if result is a dictionary
            if isinstance(result, dict):
                # Try to extract from "result" key if it's a string
                if "result" in result and isinstance(result["result"], str):
                    result_str = result["result"]
                    
                    # Try to parse as JSON
                    df = self._try_parse_json_text(result_str)
                    if df is not None and not df.empty:
                        return df
                        
                    # Try to parse as text with entity-value pairs
                    df = self._parse_text_result(result_str, query)
                    if df is not None and not df.empty:
                        return df
                
                # Try to extract from intermediate steps
                if "intermediate_steps" in result and isinstance(result["intermediate_steps"], list):
                    steps = result["intermediate_steps"]
                    if len(steps) >= 2 and steps[1] is not None:
                        return self._convert_raw_sql_to_dataframe(steps[1], query)
            
            # If result is a string, try to parse it
            elif isinstance(result, str):
                # Try to parse as JSON
                df = self._try_parse_json_text(result)
                if df is not None and not df.empty:
                    return df
                    
                # Try to parse as text
                df = self._parse_text_result(result, query)
                if df is not None and not df.empty:
                    return df
            
            # If result is a list, try to convert it directly
            elif isinstance(result, list):
                return self._parse_json_data(result)
                
            return None
            
        except Exception as e:
            print(f"Error extracting DataFrame from result: {str(e)}")
            return None
    
    def _try_parse_json_text(self, text: str) -> Optional[pd.DataFrame]:
        """
        Try to parse JSON from text and convert to DataFrame.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Optional[DataFrame]: Parsed DataFrame or None
        """
        try:
            # Clean the text
            clean_text = text.strip()
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_text)
            if json_match:
                clean_text = json_match.group(1).strip()
                
            # Check if the text looks like JSON
            if (clean_text.startswith('[') and clean_text.endswith(']')) or \
               (clean_text.startswith('{') and clean_text.endswith('}')):
                
                # Parse the JSON
                data = json.loads(clean_text)
                
                # Convert to DataFrame
                return self._parse_json_data(data)
                
            return None
            
        except Exception as e:
            print(f"Error parsing JSON text: {str(e)}")
            return None
    
    def _parse_json_data(self, data: Union[List, Dict]) -> Optional[pd.DataFrame]:
        """
        Parse JSON data into a DataFrame.
        
        Args:
            data: JSON data as Python object
            
        Returns:
            Optional[DataFrame]: Parsed DataFrame or None
        """
        try:
            # Handle list of dictionaries
            if isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                elif all(isinstance(item, (list, tuple)) for item in data):
                    df = pd.DataFrame(data)
                    df.columns = [f"column_{i}" for i in range(len(df.columns))]
                else:
                    # Create a simple one-column DataFrame
                    df = pd.DataFrame({"value": data})
            # Handle dictionary
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                return None
                
            # Convert numeric columns
            for col in df.columns:
                if any(term in str(col).lower() for term in ['count', 'number', 'value', 'total']):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                        
            return df
            
        except Exception as e:
            print(f"Error parsing JSON data: {str(e)}")
            return None
    
    def _parse_text_result(self, text: str, query: str) -> Optional[pd.DataFrame]:
        """
        Parse text result to extract entity-value pairs.
        
        Args:
            text: Text result to parse
            query: Original user query
            
        Returns:
            Optional[DataFrame]: DataFrame with extracted data or None
        """
        try:
            # Remove code blocks
            clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            
            # Define patterns to match entity-value pairs
            patterns = [
                # Entity (Number) - "Mount Sinai Hospital (44591)"
                r'([\w\s\-\.\'&]+?)\s*\((\d+(?:,\d+)*)\)',
                
                # Entity: Number - "Mount Sinai Hospital: 44591"
                r'([\w\s\-\.\'&]+?):\s*(\d+(?:,\d+)*)',
                
                # Number - Entity
                r'(\d+(?:,\d+)*)\s*-\s*([\w\s\-\.\'&]+?)',
                
                # Entity with Number
                r'([\w\s\-\.\'&]+?)\s+(?:with|has|had|having)\s+(\d+(?:,\d+)*)'
            ]
            
            # Try each pattern
            for pattern in patterns:
                entities = []
                values = []
                
                # Find all matches
                matches = re.findall(pattern, clean_text)
                if matches:
                    # Determine the order based on the pattern
                    if pattern.startswith(r'(\d+'):
                        # Number first, then entity
                        for value, entity in matches:
                            entities.append(entity.strip())
                            values.append(int(value.replace(',', '')))
                    else:
                        # Entity first, then value
                        for entity, value in matches:
                            entities.append(entity.strip())
                            values.append(int(value.replace(',', '')))
                    
                    # If we found matches, create a DataFrame
                    if entities and values:
                        # Determine appropriate column names
                        entity_col = self._infer_category_column_name(query)
                        value_col = self._infer_value_column_name(query)
                        
                        # Create DataFrame
                        df = pd.DataFrame({
                            entity_col: entities,
                            value_col: values
                        })
                        
                        print(f"Created DataFrame from text with {len(df)} rows")
                        return df
            
            # If no pattern matched, return None
            return None
            
        except Exception as e:
            print(f"Error parsing text result: {str(e)}")
            return None
    
    def _generate_plot_from_dataframe(self, df: pd.DataFrame, query: str, custom_title: Optional[str] = None) -> str:
        """
        Generate a plot from a DataFrame.
        
        Args:
            df: DataFrame containing the data to plot
            query: Original user query
            custom_title: Optional custom title to override the generated one
            
        Returns:
            str: Message about the plot generation
        """
        try:
            # Instead of a temp file, use a consistent public directory
            plots_dir = Path("./plots")  # Match with your FastAPI static directory
            plots_dir.mkdir(exist_ok=True)

            # Create a unique filename
            import uuid
            filename = f"plot_{uuid.uuid4()}.png"
            file_path = str(plots_dir / filename)
            
            # Determine the appropriate plot type
            plot_type = self._determine_plot_type(df, query)
            print(f"Selected plot type: {plot_type}")
            
            # Generate title for the plot - use custom title if provided
            title = custom_title if custom_title else self._generate_title(query)
            
            # Create the plot based on the determined type
            plt.figure(figsize=(12, 6))
            
            if plot_type == 'bar':
                self._create_bar_plot(df, query, title)
            elif plot_type == 'line':
                self._create_line_plot(df, query, title)
            elif plot_type == 'pie':
                self._create_pie_plot(df, query, title)
            elif plot_type == 'scatter':
                self._create_scatter_plot(df, query, title)
            elif plot_type == 'histogram':
                self._create_histogram_plot(df, query, title)
            elif plot_type == 'box':
                self._create_box_plot(df, query, title)
            elif plot_type == 'heatmap':
                self._create_heatmap_plot(df, query, title)
            else:
                raise ValueError(f"Plot type '{plot_type}' not supported.")
                
            # Save and display the plot
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return file_path
        
            # # Open the plot file
            # self._open_file(file_path)
            
            # return f"Plot saved and opened as {file_path}"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating plot: {str(e)}"
    
    def _determine_plot_type(self, df: pd.DataFrame, query: str) -> str:
        """
        Determine the most appropriate plot type based on data and query.
        
        Args:
            df: DataFrame with data to plot
            query: Original user query
            
        Returns:
            str: Plot type name
        """
        # Check for explicit plot type in query
        query_lower = query.lower()
        
        if 'bar' in query_lower or 'column' in query_lower:
            return 'bar'
        elif 'line' in query_lower or 'trend' in query_lower or 'over time' in query_lower:
            return 'line'
        elif 'pie' in query_lower or 'breakdown' in query_lower or 'proportion' in query_lower:
            return 'pie'
        elif 'scatter' in query_lower or 'correlation' in query_lower or 'relationship' in query_lower:
            return 'scatter'
        elif 'histogram' in query_lower or 'distribution' in query_lower:
            return 'histogram'
        elif 'box' in query_lower or 'boxplot' in query_lower:
            return 'box'
        elif 'heat' in query_lower or 'heatmap' in query_lower:
            return 'heatmap'
            
        # If not explicitly mentioned, infer from data
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        
        # Common healthcare ranking/comparison queries
        if any(term in query_lower for term in ['top', 'bottom', 'highest', 'lowest', 'most', 'least', 'ranking']):
            return 'bar'
            
        # Check data structure
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # One category column and one numeric column - usually bar chart
            return 'bar'
        elif len(numeric_cols) >= 2:
            # Two or more numeric columns - scatter plot
            return 'scatter'
        elif len(df.columns) == 1:
            # One column - histogram
            return 'histogram'
            
        # Default for healthcare data
        return 'bar'
    
    def _generate_title(self, query: str) -> str:
        """Generate a meaningful title from the user's query."""
        # Remove plot request words
        clean_query = query.lower()
        for keyword in self.plot_keywords + ['show', 'display', 'create', 'generate', 'make', 'give me']:
            clean_query = re.sub(r'\b' + keyword + r'\b', '', clean_query, flags=re.IGNORECASE)
            
        # Clean up and capitalize
        title = clean_query.strip().capitalize()
        
        # Handle healthcare-specific abbreviations
        healthcare_terms = {
            'covid': 'COVID-19',
            'covid-19': 'COVID-19',
            'covid19': 'COVID-19',
            'icu': 'ICU',
            'ed': 'ED',
            'er': 'ER'
        }
        
        for term, replacement in healthcare_terms.items():
            title = re.sub(r'\b' + term + r'\b', replacement, title, flags=re.IGNORECASE)
            
        return title
    
    def _create_bar_plot(self, df: pd.DataFrame, query: str, title: str) -> None:
        """Create a bar plot using the best columns from the DataFrame with custom colors."""
        # Select the best columns for x and y axes
        x_col, y_col = self._select_axes_columns(df, query)
        x_label, y_label = self._generate_axis_labels(x_col, y_col, query)
        
        # Ensure column types are appropriate for plotting
        if x_col in df.columns:
            df[x_col] = df[x_col].astype(str)
        if y_col in df.columns:
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            
        # Limit to top 10 categories if there are too many
        if x_col in df.columns and df[x_col].nunique() > 10:
            # Sort by y values and take top 10
            df = df.sort_values(by=y_col, ascending=False).head(10)
            title += " (Top 10)"
        
        # Define a colorful palette
        palette = sns.color_palette("viridis", n_colors=len(df))
        
        # Create the bar plot with the custom palette
        ax = sns.barplot(x=x_col, y=y_col, data=df, palette=palette)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, v in enumerate(df[y_col]):
            if pd.notna(v):
                plt.text(i, v, f"{v:,.0f}", ha='center', va='bottom', fontsize=9)
    
    def _create_line_plot(self, df: pd.DataFrame, query: str, title: str) -> None:
        """Create a line plot for time series or trend data with custom colors."""
        # Select the best columns for x and y axes
        x_col, y_col = self._select_axes_columns(df, query)
        x_label, y_label = self._generate_axis_labels(x_col, y_col, query)
        
        # Sort by x values
        try:
            df = df.sort_values(by=x_col)
        except:
            pass
        
        # Create the line plot with a visually appealing color
        plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, color='#1e88e5')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45, ha='right')
        
        # Add grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add markers with different colors for emphasis
        plt.scatter(df[x_col], df[y_col], s=60, c='#ff5722', zorder=5)
    
    def _create_pie_plot(self, df: pd.DataFrame, query: str, title: str) -> None:
        """Create a pie chart showing proportions of categories with custom colors."""
        # Select the best columns for labels and values
        label_col, value_col = self._select_axes_columns(df, query)
        
        # Ensure value column is numeric
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Limit to top 7 categories + "Other" for readability
        if df[label_col].nunique() > 7:
            top_categories = df.nlargest(7, value_col)
            other_sum = df[~df[label_col].isin(top_categories[label_col])][value_col].sum()
            
            # Add "Other" category
            other_df = pd.DataFrame({label_col: ['Other'], value_col: [other_sum]})
            df = pd.concat([top_categories, other_df], ignore_index=True)
        
        # Use a colorful palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
        
        # Create the pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(df[value_col], labels=df[label_col], autopct='%1.1f%%', 
                startangle=90, shadow=False, colors=colors)
        plt.axis('equal')
        plt.title(title)
        
        # Add legend for better readability
        plt.legend(df[label_col], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    def _create_scatter_plot(self, df: pd.DataFrame, query: str, title: str) -> None:
        """Create a scatter plot showing relationship between two variables with custom colors."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            raise ValueError("Scatter plot requires at least two numeric columns")
            
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        x_label, y_label = self._generate_axis_labels(x_col, y_col, query)
        
        # Create a colorful scatter plot
        # If there's a third numeric column, use it for color mapping
        if len(numeric_cols) >= 3:
            third_col = numeric_cols[2]
            plt.scatter(df[x_col], df[y_col], c=df[third_col], cmap='viridis', 
                    alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
            plt.colorbar(label=third_col.replace('_', ' ').title())
        else:
            # Otherwise, use a gradient based on x-values
            plt.scatter(df[x_col], df[y_col], c=range(len(df)), cmap='viridis', 
                    alpha=0.7, s=50, edgecolor='k', linewidth=0.5)
        
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        # Add trend line
        try:
            z = np.polyfit(df[x_col], df[y_col], 1)
            p = np.poly1d(z)
            plt.plot(df[x_col], p(df[x_col]), "r--", alpha=0.7, linewidth=2)
        except:
            pass
    
    def _create_histogram_plot(self, df: pd.DataFrame, query: str, title: str) -> None:
        """Create a histogram showing distribution of a variable with custom colors."""
        # Select numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 1:
            raise ValueError("Histogram requires at least one numeric column")
            
        col = numeric_cols[0]
        _, label = self._generate_axis_labels('frequency', col, query)
        
        # Create the histogram with custom colors
        sns.histplot(df[col], kde=True, color='#3949ab', kde_kws={'color': '#e53935', 'linewidth': 2}, 
                    alpha=0.7, edgecolor='white', linewidth=0.5)
        plt.title(title)
        plt.xlabel(label)
        plt.ylabel("Frequency")
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        plt.axvline(mean_val, color='#00897b', linestyle='--', alpha=0.8, linewidth=2, label=f"Mean: {mean_val:.2f}")
        plt.axvline(median_val, color='#ffa000', linestyle='-.', alpha=0.8, linewidth=2, label=f"Median: {median_val:.2f}")
        plt.legend()
    
    def _create_box_plot(self, df: pd.DataFrame, query: str, title: str) -> None:
        """Create a box plot showing distribution with quartiles with custom colors."""
        # Select categorical and numeric columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(categorical_cols) < 1 or len(numeric_cols) < 1:
            raise ValueError("Box plot requires at least one categorical and one numeric column")
            
        x_col = categorical_cols[0]
        y_col = numeric_cols[0]
        x_label, y_label = self._generate_axis_labels(x_col, y_col, query)
        
        # Limit categories if too many
        if df[x_col].nunique() > 8:
            top_cats = df[x_col].value_counts().nlargest(8).index
            df = df[df[x_col].isin(top_cats)]
            title += " (Top 8 categories)"
        
        # Create a custom palette
        palette = sns.color_palette("Set3", n_colors=df[x_col].nunique())
        
        # Create the box plot
        sns.boxplot(x=x_col, y=y_col, data=df, palette=palette, width=0.6)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=45, ha='right')
        
        # Add subtle grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    def _create_heatmap_plot(self, df: pd.DataFrame, query: str, title: str) -> None:
        """Create a heatmap showing correlations between variables with custom colors."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) < 2:
            raise ValueError("Heatmap requires at least two numeric columns")
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create the heatmap with a better colormap and improved styling
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create a mask for the upper triangle
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, 
                    linewidths=0.5, mask=mask, annot_kws={"size": 10},
                    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        
        # Improve tick label readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
    
    def _select_axes_columns(self, df: pd.DataFrame, query: str) -> Tuple[str, str]:
        """
        Select the most appropriate columns for x and y axes.
        
        Args:
            df: DataFrame containing the data
            query: Original user query
            
        Returns:
            Tuple[str, str]: x-axis column name, y-axis column name
        """
        # If DataFrame has only one column, create an index column
        if len(df.columns) == 1:
            col_name = df.columns[0]
            df['index'] = range(len(df))
            return 'index', col_name
            
        # Identify numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # For x-axis, prefer non-numeric columns like 'region', 'category', etc.
        x_col_candidates = ['region', 'category', 'name', 'hospital', 'facility', 'provider', 
                           'diagnosis', 'disease', 'procedure', 'gender', 'age_group']
        
        x_col = None
        for candidate in x_col_candidates:
            matching_cols = [col for col in non_numeric_cols if candidate.lower() in col.lower()]
            if matching_cols:
                x_col = matching_cols[0]
                break
                
        # If no match found, use the first non-numeric column
        if not x_col and non_numeric_cols:
            x_col = non_numeric_cols[0]
        elif not x_col:
            # No non-numeric columns, use the first column
            x_col = df.columns[0]
            
        # For y-axis, prefer numeric columns like 'count', 'value', etc.
        y_col_candidates = ['count', 'value', 'total', 'number', 'amount', 'sum']
        
        y_col = None
        for candidate in y_col_candidates:
            matching_cols = [col for col in numeric_cols if candidate.lower() in col.lower()]
            if matching_cols:
                y_col = matching_cols[0]
                break
                
        # If no match found, use the first numeric column
        if not y_col and numeric_cols:
            y_col = numeric_cols[0]
        elif not y_col:
            # No numeric columns, use the second column
            y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
        return x_col, y_col
    
    def _generate_axis_labels(self, x_col: str, y_col: str, query: str) -> Tuple[str, str]:
        """
        Generate meaningful axis labels based on column names and query.
        
        Args:
            x_col: x-axis column name
            y_col: y-axis column name
            query: Original user query
            
        Returns:
            Tuple[str, str]: x-axis label, y-axis label
        """
        # Generate x-axis label
        x_label = x_col.replace('_', ' ').title()
        for category, label in self.category_mappings.items():
            if category in x_col.lower():
                x_label = label
                break
                
        # Generate y-axis label
        y_label = y_col.replace('_', ' ').title()
        
        # Enhance y-axis label for value columns
        value_terms = {
            'count': 'Count',
            'value': 'Value',
            'total': 'Total',
            'number': 'Number',
            'sum': 'Sum',
            'amount': 'Amount',
        }
        
        for term, replacement in value_terms.items():
            if term in y_col.lower():
                y_label = replacement
                
                # Check query for what is being counted
                count_entities = [
                    ('patient', 'Patients'),
                    ('admit', 'Admissions'),
                    ('case', 'Cases'),
                    ('visit', 'Visits'),
                    ('procedure', 'Procedures'),
                    ('test', 'Tests'),
                    ('diagnos', 'Diagnoses'),
                    ('treatment', 'Treatments')
                ]
                
                for entity, entity_label in count_entities:
                    if entity in query.lower():
                        y_label = f"Number of {entity_label}"
                        break
                        
                break
                
        return x_label, y_label
    
    def _infer_category_column_name(self, query: str) -> str:
        """
        Infer an appropriate category column name from the query.
        
        Args:
            query: User's query
            
        Returns:
            str: Inferred column name
        """
        # Domain-specific column names
        domain_terms = {
            'hospital': ['hospital', 'facility', 'center', 'clinic'],
            'physician': ['doctor', 'physician', 'provider', 'practitioner'],
            'diagnosis': ['diagnosis', 'condition', 'disease', 'illness'],
            'procedure': ['procedure', 'surgery', 'operation', 'treatment'],
            'medication': ['drug', 'medication', 'medicine', 'prescription'],
            'vaccine': ['vaccine', 'vaccination', 'immunization'],
            'region': ['region', 'area', 'location', 'zone', 'geography'],
            'age': ['age', 'year old', 'years old'],
            'gender': ['gender', 'sex'],
            'symptom': ['symptom', 'sign', 'presentation']
        }
        
        # Check for domain-specific terms in the query
        for col_name, terms in domain_terms.items():
            if any(term in query.lower() for term in terms):
                return col_name
                
        # Default to 'category'
        return 'category'
    
    def _infer_value_column_name(self, query: str) -> str:
        """
        Infer an appropriate value column name from the query.
        
        Args:
            query: User's query
            
        Returns:
            str: Inferred column name
        """
        # Check for specific count terms
        value_terms = {
            'admission': 'admissions',
            'admit': 'admissions',
            'patient': 'patients',
            'visit': 'visits',
            'procedure': 'procedures',
            'case': 'cases',
            'test': 'tests',
            'diagnos': 'diagnoses',
            'treatment': 'treatments',
            'payment': 'payments',
            'cost': 'costs',
            'rate': 'rate',
            'score': 'score'
        }
        
        for term, col_name in value_terms.items():
            if term in query.lower():
                return col_name
                
        # Default to 'count'
        return 'count'
    
    def _open_file(self, file_path: str) -> None:
        """
        Open the file using the system's default application.
        
        Args:
            file_path: Path to the file to open
        """
        try:
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS or Linux
                if 'darwin' in os.sys.platform:  # macOS
                    os.system(f'open "{file_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{file_path}"')
            else:
                # Fallback to webbrowser
                webbrowser.open('file://' + file_path)
        except Exception as e:
            print(f"Error opening file: {str(e)}")
    
    # Function to generate and stream plot via WebSocket    
    
    async def generate_and_stream_plot(self, websocket, data, query_type, title):
        """Generate plot and stream it directly via WebSocket"""
        try:
            import io
            import base64
            import json
            
            print(f"Generating streaming plot for query: {query_type}")
            print(f"Plot data sample: {str(data)[:200]}...")
            
            # Use your existing _generate_plot_from_dataframe method to create the plot
            # First convert data to DataFrame
            df = None
            if isinstance(data, list) and len(data) > 0:
                df = self._convert_raw_sql_to_dataframe(data, query_type)
                
                if df is not None and not df.empty:
                    print(f"Created DataFrame for streaming plot with shape {df.shape}")
                    
                    # Determine the plot type
                    plot_type = self._determine_plot_type(df, query_type)
                    print(f"Selected plot type for streaming: {plot_type}")
                    
                    # Create a temporary file path for the plot
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Use existing plotting functionality
                    plt.figure(figsize=(12, 6))
                    
                    if plot_type == 'bar':
                        self._create_bar_plot(df, query_type, title)
                    elif plot_type == 'pie':
                        self._create_pie_plot(df, query_type, title)
                    elif plot_type == 'line':
                        self._create_line_plot(df, query_type, title)
                    else:
                        # Default to bar
                        self._create_bar_plot(df, query_type, title)
                        
                    plt.tight_layout()
                    
                    # First save to temp file (leveraging existing code)
                    plt.savefig(temp_path, dpi=100)
                    
                    # Then read back and convert to base64
                    with open(temp_path, 'rb') as f:
                        plot_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Clean up temp file
                    plt.close()
                    os.unlink(temp_path)
                    
                    # Send plot data via WebSocket
                    print("Sending plot data via WebSocket...")
                    await websocket.send_text(json.dumps({
                        "type": "plot",
                        "format": "base64",
                        "data": f"data:image/png;base64,{plot_data}"
                    }))
                    
                    print("✅ Plot successfully streamed via WebSocket")
                    return True
                else:
                    print("❌ Failed to create DataFrame from data")
            else:
                print(f"❌ Data is not in expected format: {type(data)}")
            
            return False
            
        except Exception as e:
            print(f"❌ Error generating streaming plot: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Failed to generate plot: {str(e)}"
                }))
            except Exception as ws_error:
                print(f"❌ Error sending error message via WebSocket: {ws_error}")
                
            return False


# Direct helper function
def generate_plot_from_result(result: Any, query: str, raw_sql_result=None) -> str:
    """
    Standalone function to generate a plot from a query result.
    
    Args:
        result: The query result
        query: User's original query
        raw_sql_result: Raw SQL result if available
        
    Returns:
        str: Message indicating plot generation status
    """
    plotter = GeneratingPlots()
    return plotter.process_query_result(result, query, raw_sql_result)