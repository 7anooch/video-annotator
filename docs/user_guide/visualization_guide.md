# Visualization Guide

This guide provides detailed information about visualizing annotations using the Video Annotator application.

## Visualization Tools

The Video Annotator provides several tools for visualizing annotations:

1. **Ethogram**: A timeline view of annotations showing when each behavior occurs
2. **Label Distribution**: A bar chart showing the frequency of each label
3. **Label Timeline**: A scatter plot showing when each label occurs throughout the video
4. **Transition Matrix**: A heatmap showing how often one behavior transitions to another

## Launching the Visualization Tool

You can launch the visualization tool using one of the following methods:

### Using the Run Scripts

```bash
# On Windows
scripts\run.bat

# On macOS/Linux
./scripts/run.sh
```

Select "8. Run Visualization Tool" from the menu.

### Using the Command Line

```bash
# Run the visualization tool
python main.py --mode visualize
```

## Using the Visualization Tool

### 1. Select an Annotation File

When you launch the visualization tool, you'll need to select an annotation file (CSV) to visualize.

1. Click the "Browse" button
2. Navigate to your annotation file
3. Select the file and click "Open"

![Select Annotation File](../images/select_annotation_file.png)

### 2. Choose Visualization Options

The visualization tool allows you to choose which visualizations to display:

- **Ethogram**: A timeline view of annotations
- **Label Distribution**: A bar chart of label frequencies
- **Label Timeline**: A scatter plot of label occurrences
- **Transition Matrix**: A heatmap of behavior transitions

Select the checkboxes for the visualizations you want to display.

![Visualization Options](../images/visualization_options.png)

### 3. Generate Visualizations

Click the "Visualize" button to generate the selected visualizations.

![Generate Visualizations](../images/generate_visualizations.png)

## Understanding the Visualizations

### Ethogram

The ethogram provides a timeline view of annotations, showing when each behavior occurs throughout the video.

- **X-axis**: Frame number
- **Y-axis**: None (the entire vertical space represents the current behavior)
- **Colors**: Each color represents a different label (e.g., red for Stop, green for Run, blue for Turn)

![Ethogram](../images/ethogram.png)

#### Interpreting the Ethogram

- **Continuous blocks of color** indicate periods where the same behavior persists
- **Transitions between colors** indicate changes in behavior
- **Patterns in the ethogram** can reveal behavioral sequences or rhythms

### Label Distribution

The label distribution shows the frequency of each label in the annotations.

- **X-axis**: Label names
- **Y-axis**: Count (number of frames with each label)
- **Colors**: Each bar is colored according to the label it represents

![Label Distribution](../images/label_distribution.png)

#### Interpreting the Label Distribution

- **Relative heights of bars** indicate the prevalence of each behavior
- **Dominant behaviors** have taller bars
- **Rare behaviors** have shorter bars

### Label Timeline

The label timeline shows when each label occurs throughout the video.

- **X-axis**: Frame number
- **Y-axis**: Label names
- **Points**: Each point represents a frame with the corresponding label

![Label Timeline](../images/label_timeline.png)

#### Interpreting the Label Timeline

- **Clusters of points** indicate periods where a behavior occurs frequently
- **Gaps** indicate periods where a behavior is absent
- **Alternating patterns** can reveal behavioral cycles

### Transition Matrix

The transition matrix shows how often one behavior transitions to another.

- **X-axis**: To label (the behavior being transitioned to)
- **Y-axis**: From label (the behavior being transitioned from)
- **Colors**: Intensity of color indicates transition probability (darker = higher probability)
- **Numbers**: Transition probabilities (0-1)

![Transition Matrix](../images/transition_matrix.png)

#### Interpreting the Transition Matrix

- **Diagonal elements** represent the probability of staying in the same behavior
- **Off-diagonal elements** represent the probability of transitioning to a different behavior
- **High probabilities** (darker colors) indicate common transitions
- **Low probabilities** (lighter colors) indicate rare transitions

## Saving Visualizations

You can save the visualizations by right-clicking on the figure and selecting "Save as" or by using the save button in the matplotlib toolbar.

## Customizing Visualizations

The visualizations use the colors defined in your configuration file for each label. You can customize these colors using the configuration editor. See the [Configuration Guide](configuration_guide.md) for more information.

## Advanced Visualization

### Comparing Multiple Annotations

To compare annotations from multiple annotators, you can use the plot tool:

```bash
# Using the run scripts
# Select "4. Run Plot Tool" from the menu

# Using the command line
python main.py --mode plot
```

This will allow you to select multiple annotation files and display them as ethograms for comparison.

![Compare Annotations](../images/compare_annotations.png)

### Analyzing Annotation Agreement

To analyze the agreement between multiple annotators, you can use the analysis tool:

```bash
# Using the run scripts
# Select "5. Run Analysis Tool" from the menu

# Using the command line
python main.py --mode analyze
```

This will calculate various agreement metrics between annotators.

![Analyze Agreement](../images/analyze_agreement.png)

## Next Steps

After visualizing your annotations, you can:

- [Analyze](analysis_guide.md) your annotations to extract insights
- [Export](export_guide.md) your annotations for use in other applications
- [Configure](configuration_guide.md) the application to customize the visualizations
