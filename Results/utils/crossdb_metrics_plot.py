import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import io
import logging
import os
import re
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def chart():
    fer_inf_data = """Model, Model size(mbs), Avg Inference Time (ms)
    EfficientNetB0, 15.6, 6.621059998724377
    EfficientNetV2, 203, 26.823309999599587
    Eva, 1120, 10.772694998740917
    Eva02, 327, 10.91950500049279
    Eva02_wide, 981, 9.603865000099177
    FBNetV3B, 25.6, 11.774190002324758
    MobileNetV3, 16.2, 5.776675000379328
    MobileNetV4, 140, 16.19050000081188
    ResEmoteNet, 306, 3.516404998663347
    ResNeSt, 97.4, 13.729989999410463
    ResNeXt50, 88, 5.654374999721767
    VGG19, 532, 5.504244998883223
    Ensemble_Eva02_Eva_Res_EffV2, 1956, 52.24735000001965
    """

    test_bench_data = """Model,Accuracy,Weighted Precision,Weighted Recall,Weighted F1
    EfficientNetB0,0.6112040133779264,0.6148167694180142,0.6112040133779264,0.6049205906972677
    EfficientNetV2,0.7683946488294314,0.7762990747892783,0.7683946488294314,0.7697228803593507
    Eva,0.7923634336677815,0.7980166406256001,0.7923634336677815,0.7921075709567487
    Eva02,0.8455964325529542,0.8520601469251473,0.8455964325529542,0.8432994817049176
    Eva02_wide,0.8815063473367361,0.8821628538091995,0.8815063473367361,0.8811620042681228
    FBNetV3B,0.6984392419175028,0.7066859528641578,0.6984392419175028,0.6965238607635135
    MobileNetV3,0.6613712374581939,0.6680181122120258,0.6613712374581939,0.6610375599869943
    MobileNetV4,0.6079710144927537,0.5896544561955012,0.59384615384615385,0.5975057773916534
    ResEmoteNet,0.8294314381270903,0.8360860694377413,0.8294314381270903,0.8291876749086593
    ResNeSt,0.7455406911928651,0.7542896831371709,0.7455406911928651,0.746494106166825
    ResNeXt50,0.6638795986622074,0.6691087325402402,0.6638795986622074,0.6576968379739313
    VGG19,0.6524526198439242,0.671529208213851,0.6524526198439242,0.6683319609107572
    Ensemble_Eva02_Eva_Res_EffV2,0.9013377926421404,0.9038919035202931,0.9013377926421404,0.900721479316398
    """

    # or give path to csv file here
    fer_inf_filename = io.StringIO(fer_inf_data)
    test_bench_filename = io.StringIO(test_bench_data)

    plot_title = "FER Model Benchmark: Performance vs. Speed"
    x_axis_label = "Average Inference Time (ms) - Lower is Better"
    y_axis_label = "Weighted F1-Score - Higher is Better"
    bubble_size_scale = 1.5
    min_bubble_size = 30
    color_map = 'jet'
    output_plot_filename = 'fer_bubble_perf_speed_no_size_legend.png'
    annotate_models = True

    def clean_col_names(df):
        cols = df.columns; cols = cols.str.strip(); cols = cols.str.lower()
        cols = cols.str.replace(' ', '_', regex=False)
        cols = cols.str.replace(r'[\(\)]', '', regex=True)
        cols = cols.str.replace(r'[?/]', '_', regex=True)
        cols = cols.str.replace('%', 'percent', regex=False)
        df.columns = cols
        return df

    def standardize_model_name(name):
        name = name.strip()
        name = re.sub(r'^MyEnsemble', 'Ensemble', name, flags=re.IGNORECASE)
        return name

    try:
        df_inf = pd.read_csv(fer_inf_filename, skipinitialspace=True).pipe(clean_col_names)
        df_perf = pd.read_csv(test_bench_filename, skipinitialspace=True).pipe(clean_col_names)
        logging.info(f"Read {len(df_inf)} inference records and {len(df_perf)} performance records.")

        required_inf_cols = {'model', 'model_sizembs', 'avg_inference_time_ms'}
        required_perf_cols = {'model', 'accuracy', 'weighted_f1'}
        if not required_inf_cols.issubset(df_inf.columns): raise ValueError(f"Missing required columns in inference data.")
        if not required_perf_cols.issubset(df_perf.columns): raise ValueError(f"Missing required columns in performance data.")

        df_inf['model'] = df_inf['model'].apply(standardize_model_name)
        df_perf['model'] = df_perf['model'].apply(standardize_model_name)

        numeric_cols_inf = ['model_sizembs', 'avg_inference_time_ms']
        numeric_cols_perf = ['accuracy', 'weighted_f1']
        for col in numeric_cols_inf: df_inf[col] = pd.to_numeric(df_inf[col], errors='coerce')
        for col in numeric_cols_perf: df_perf[col] = pd.to_numeric(df_perf[col], errors='coerce')
        df_inf.dropna(subset=numeric_cols_inf, inplace=True)
        df_perf.dropna(subset=numeric_cols_perf, inplace=True)

        df_merged = pd.merge(df_perf, df_inf, on='model', how='inner')
        if df_merged.empty: raise ValueError("Merge resulted in an empty DataFrame.")
        logging.info(f"Successfully merged data for {len(df_merged)} models.")

        model_size_col = 'model_sizembs'
        if model_size_col in df_merged.columns and df_merged[model_size_col].nunique() > 1:
            max_size = df_merged[model_size_col].max()
            min_size = df_merged[model_size_col].min()
            df_merged['bubble_size'] = min_bubble_size + ((df_merged[model_size_col] - min_size) / (max_size - min_size)) * (max_size * bubble_size_scale)
        elif model_size_col in df_merged.columns:
            logging.warning(f"Only one unique model size found or calculation issue. Using default size.")
            df_merged['bubble_size'] = min_bubble_size * bubble_size_scale * 2
        else:
            logging.error(f"Model size column '{model_size_col}' not found for bubble sizing. Using default size.")
            df_merged['bubble_size'] = min_bubble_size * bubble_size_scale * 2

        plt.style.use('seaborn-v0_8-notebook')
        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            df_merged['avg_inference_time_ms'],
            df_merged['weighted_f1'],
            s=df_merged['bubble_size'],       
            c=df_merged['weighted_f1'],       
            cmap=color_map,
            alpha=0.7,                         
            edgecolors='k',
            linewidth=0.6,
            zorder=2
        )

        if annotate_models:
            texts = []
            for i, row in df_merged.iterrows():
                texts.append(ax.text(row['avg_inference_time_ms'] * 1.015,
                                    row['weighted_f1'],
                                    row['model'],
                                    fontsize=9,
                                    zorder=3))
            try:
                from adjustText import adjust_text
                adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                logging.info("Used adjustText to prevent label overlap.")
            except ImportError:
                logging.warning("Library 'adjustText' not found. Labels might overlap. Install with: pip install adjustText")
                pass

        ax.set_title(plot_title, fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel(x_axis_label, fontsize=12)
        ax.set_ylabel(y_axis_label, fontsize=12)

        cbar = fig.colorbar(scatter)
        cbar.set_label('Weighted F1-Score', fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

        ax.grid(True, linestyle='--', alpha=0.6, zorder=1)

        plt.tight_layout(pad=1.5)

        if output_plot_filename:
            try:
                plt.savefig(output_plot_filename, dpi=500, bbox_inches='tight')
                logging.info(f"Bubble benchmark plot (no size legend) saved to '{output_plot_filename}'")
            except Exception as e:
                logging.error(f"Failed to save plot: {e}")

        plt.show()

    except FileNotFoundError:
        logging.error(f"Error: One or both CSV files not found. Check paths.")
    except ValueError as ve:
        logging.error(f"Data Error: {ve}")
    except KeyError as ke:
        logging.error(f"Column Error: A required column name ({ke}) might be incorrect after cleaning/merging.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

def table():
    fer_inf_data = """Model, Model size(mbs), Avg Inference Time (ms)
    EfficientNetB0, 15.6, 6.621059998724377
    EfficientNetV2, 203, 26.823309999599587
    Eva, 1120, 10.772694998740917
    Eva02, 327, 10.91950500049279
    Eva02_wide, 981, 9.603865000099177
    FBNetV3B, 25.6, 11.774190002324758
    MobileNetV3, 16.2, 5.776675000379328
    MobileNetV4, 140, 16.19050000081188
    ResEmoteNet, 306, 3.516404998663347
    ResNeSt, 97.4, 13.729989999410463
    ResNeXt50, 88, 5.654374999721767
    VGG19, 532, 5.504244998883223
    Ensemble_Eva02_Eva_Res_EffV2, 1956, 52.24735000001965
    """

    test_bench_data = """Model,Accuracy,Weighted Precision,Weighted Recall,Weighted F1
    EfficientNetB0,0.6112040133779264,0.6148167694180142,0.6112040133779264,0.6049205906972677
    EfficientNetV2,0.7683946488294314,0.7762990747892783,0.7683946488294314,0.7697228803593507
    Eva,0.7923634336677815,0.7980166406256001,0.7923634336677815,0.7921075709567487
    Eva02,0.8455964325529542,0.8520601469251473,0.8455964325529542,0.8432994817049176
    Eva02_wide,0.8815063473367361,0.8821628538091995,0.8815063473367361,0.8811620042681228
    FBNetV3B,0.6984392419175028,0.7066859528641578,0.6984392419175028,0.6965238607635135
    MobileNetV3,0.6613712374581939,0.6680181122120258,0.6613712374581939,0.6610375599869943
    MobileNetV4,0.6079710144927537,0.5896544561955012,0.59384615384615385,0.5975057773916534
    ResEmoteNet,0.8294314381270903,0.8360860694377413,0.8294314381270903,0.8291876749086593
    ResNeSt,0.7455406911928651,0.7542896831371709,0.7455406911928651,0.746494106166825
    ResNeXt50,0.6638795986622074,0.6691087325402402,0.6638795986622074,0.6576968379739313
    VGG19,0.6524526198439242,0.671529208213851,0.6524526198439242,0.6683319609107572
    Ensemble_Eva02_Eva_Res_EffV2,0.9013377926421404,0.9038919035202931,0.9013377926421404,0.900721479316398
    """

    fer_inf_filename = io.StringIO(fer_inf_data)
    test_bench_filename = io.StringIO(test_bench_data)
    # -------------------------------------------------

    # --- Configuration ---
    sort_by_column = 'avg_inference_time_ms' # 'model', 'model_sizembs', 'avg_inference_time_ms'
    sort_ascending = True
    table_format_console = 'pretty'
    table_format_markdown = 'github'

    output_dir = "tables"
    output_console_filename = "model_summary_console.txt"
    output_markdown_filename = "model_summary.md"
    # ----------------------------------------

    def clean_col_names(df):
        cols = df.columns; cols = cols.str.strip(); cols = cols.str.lower()
        cols = cols.str.replace(' ', '_', regex=False)
        cols = cols.str.replace(r'[\(\)]', '', regex=True)
        cols = cols.str.replace(r'[?/]', '_', regex=True)
        cols = cols.str.replace('%', 'percent', regex=False)
        df.columns = cols
        return df

    def standardize_model_name(name):
        name = name.strip()
        name = re.sub(r'^MyEnsemble', 'Ensemble', name, flags=re.IGNORECASE)
        return name

    try:
        df_inf = pd.read_csv(fer_inf_filename, skipinitialspace=True).pipe(clean_col_names)
        df_perf = pd.read_csv(test_bench_filename, skipinitialspace=True).pipe(clean_col_names)
        logging.info(f"Read {len(df_inf)} inference records and {len(df_perf)} performance records.")

        required_inf_cols = {'model', 'model_sizembs', 'avg_inference_time_ms'}
        required_perf_cols = {'model', 'weighted_f1'}
        if not required_inf_cols.issubset(df_inf.columns): raise ValueError("Missing required columns in inference data.")
        if not required_perf_cols.issubset(df_perf.columns): raise ValueError("Missing required columns in performance data.")

        df_inf['model'] = df_inf['model'].apply(standardize_model_name)
        df_perf['model'] = df_perf['model'].apply(standardize_model_name)

        numeric_cols_inf = ['model_sizembs', 'avg_inference_time_ms']
        for col in numeric_cols_inf: df_inf[col] = pd.to_numeric(df_inf[col], errors='coerce')
        df_inf.dropna(subset=numeric_cols_inf, inplace=True)

        df_merged = pd.merge(df_perf[['model']], df_inf, on='model', how='inner')
        if df_merged.empty: raise ValueError("Merge resulted in an empty DataFrame.")
        logging.info(f"Successfully merged data for {len(df_merged)} models.")

        columns_for_table = ['model', 'model_sizembs', 'avg_inference_time_ms']
        df_table = df_merged[columns_for_table].copy()

        if sort_by_column in df_table.columns:
            df_table.sort_values(by=sort_by_column, ascending=sort_ascending, inplace=True)
            logging.info(f"Table sorted by '{sort_by_column}' ({'ascending' if sort_ascending else 'descending'}).")
        else:
            logging.warning(f"Column '{sort_by_column}' not found for sorting. Table remains unsorted.")

        df_table['model_sizembs'] = df_table['model_sizembs'].round(1)
        df_table['avg_inference_time_ms'] = df_table['avg_inference_time_ms'].round(2)

        headers = {'model': 'Model','model_sizembs': 'Size (MB)','avg_inference_time_ms': 'Avg Inference (ms)'}
        df_table.rename(columns=headers, inplace=True)

        console_table = tabulate(df_table, headers='keys', tablefmt=table_format_console, showindex=False, numalign="right", stralign="left")
        markdown_table = tabulate(df_table, headers='keys', tablefmt=table_format_markdown, showindex=False, numalign="right", stralign="left")

        print("\n--- Model Size and Inference Time ---")
        print(f"(Sorted by: {headers.get(sort_by_column, sort_by_column)} {'Asc' if sort_ascending else 'Desc'})")
        print("\nConsole Output Table Preview:")
        print(console_table)

        if output_console_filename or output_markdown_filename:
            try:
                os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Output directory '{output_dir}' ensured.")
            except OSError as e:
                logging.error(f"Could not create output directory '{output_dir}': {e}")
                output_console_filename = None
                output_markdown_filename = None

        if output_console_filename:
            filepath = os.path.join(output_dir, output_console_filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Model Size and Inference Time\n")
                    f.write(f"(Sorted by: {headers.get(sort_by_column, sort_by_column)} {'Asc' if sort_ascending else 'Desc'})\n\n")
                    f.write(console_table)
                logging.info(f"Console-formatted table saved successfully to: {filepath}")
            except (IOError, OSError) as e:
                logging.error(f"Failed to save console table to {filepath}: {e}")

        if output_markdown_filename:
            filepath = os.path.join(output_dir, output_markdown_filename)
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Model Size and Inference Time\n\n")
                    f.write(f"*Sorted by: {headers.get(sort_by_column, sort_by_column)} ({'Ascending' if sort_ascending else 'Descending'})*\n\n")
                    f.write(markdown_table)
                logging.info(f"Markdown table saved successfully to: {filepath}")
            except (IOError, OSError) as e:
                logging.error(f"Failed to save Markdown table to {filepath}: {e}")

    except FileNotFoundError:
        logging.error(f"Error: One or both CSV files not found.")
    except ValueError as ve:
        logging.error(f"Data Error: {ve}")
    except KeyError as ke:
        logging.error(f"Column Error: Required column name ({ke}) missing.")
    except ImportError:
        logging.error("Error: The 'tabulate' library is required. Please install it: pip install tabulate")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

# chart()
# table()
