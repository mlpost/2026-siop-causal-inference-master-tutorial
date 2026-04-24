# Use ggdag to create a causal DAG for the manager training study

# Practice here: https://cbdrh.shinyapps.io/daggle/
# Mark Hanly, Bronwyn K Brew, Anna Austin, Louisa Jorm, Software Application Profile: 
# The daggle app—a tool to support learning and teaching the graphical rules of selecting adjustment variables 
# using directed acyclic graphs, International Journal of Epidemiology, 2023;, dyad038 https://doi.org/10.1093/ije/dyad038

# outcomes: Manager Efficacy, Workload, Stay Intention, Retention
# treatment: Manager Training
# confounders: Gender, age, tenure, performance rating, organization, job family, region, num. direct reports, baseline survey scores, promotion intensity (driven by organization)
# all confounders -> treatment and outcome
# organization in particular -> promotion intensity -> treatment

# Load required libraries
library(dagitty)
library(ggdag)
library(tidyverse) 
library(ggplot2)

# Define node coordinates (coord_df method per ggdag manual)
# Layout: left-to-right causal flow; confounders on left, treatment center, outcome right
coord_df <- data.frame(
  name = c(
    "employee_demographics", 
    "performance_career", 
    "role_structure",
    "baseline_scores", 
    "organization", 
    "promotion_intensity",
    "manager_training", 
    "outcomes"
  ),
  x = c(1.5, 1.25, 1.25, 1.5, 1.75, 3, 5, 5),
  y = c(2, 1, 0, -1, -1.5, -1.5, -0.5, 0.5)
)

# Define the DAG structure using dagify()
# Note: ggdag uses ~ syntax where outcome ~ exposure means exposure -> outcome
# Define the DAG structure with collapsed outcomes
manager_training_dag <- dagify(
  # Treatment effect on outcomes (collapsed)
  outcomes ~ manager_training,
  
  # Organization drives promotion intensity (instrument-like encouragement: Z -> T only)
  promotion_intensity ~ organization,
  manager_training ~ promotion_intensity,
  
  # Organization direct effects on outcomes
  outcomes ~ organization,
  
  # All confounder groups affect treatment and outcomes
  # Employee Demographics
  manager_training ~ employee_demographics,
  outcomes ~ employee_demographics,
  
  # Performance & Career
  manager_training ~ performance_career,
  outcomes ~ performance_career,
  
  # Role & Structure
  manager_training ~ role_structure,
  outcomes ~ role_structure,
  
  # Baseline Survey Scores
  manager_training ~ baseline_scores,
  outcomes ~ baseline_scores,

  

  
  # Define node labels for display
  labels = c(
    manager_training = "Manager Training\n(T)",
    outcomes = "Outcomes\n(Efficacy, Workload,\nStay Intention, Retention)\n(Y)",
    organization = "Organization\n(X)",
    promotion_intensity = "Promotion\nIntensity\n(Z)",
    employee_demographics = "Employee\nDemographics\n(X)",
    performance_career = "Performance\n& Career\n(X)",
    role_structure = "Role &\nStructure\n(X)",
    baseline_scores = "Baseline\nSurvey Scores\n(X)"
  ),
  
  # Define exposure and outcome for highlighting
  exposure = "manager_training",
  outcome = "outcomes",

  # Explicit node coordinates (coord_df method)
  coords = coord_df
)

# Role mapping for causal inference legend:
# - Treatment: intervention whose effect we estimate
# - Outcome: primary endpoint(s)
# - Confounder: causes both treatment and outcome; may bias estimates if unadjusted
  # - Instrument-like (Z): pre-treatment org-level encouragement; Z -> T only (not on T -> Y path)
role_mapping <- c(
  manager_training = "Treatment (Exposure)",
  outcomes = "Outcome",
  promotion_intensity = "Instrument-like (Z)",
  organization = "Adjusted Confounder",
  employee_demographics = "Adjusted Confounder",
  performance_career = "Adjusted Confounder",
  role_structure = "Adjusted Confounder",
  baseline_scores = "Adjusted Confounder"
)
role_colors <- c(
  "Treatment (Exposure)" = "#2E86AB",
  "Outcome" = "#A23B72",
  "Instrument-like (Z)" = "#F4A261",
  "Adjusted Confounder" = "#6B7280"
)
role_shapes <- c(
  "Treatment (Exposure)" = 19,
  "Outcome" = 19,
  "Instrument-like (Z)" = 17,
  "Adjusted Confounder" = 15
)

# Create the main DAG visualization (per ggdag vignette)
dag_plot <- manager_training_dag %>%
  tidy_dagitty() %>%
  mutate(role = recode(name, !!!role_mapping)) %>%
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_edges_link(
    edge_color = "grey60",
    edge_width = 0.8,
    arrow = grid::arrow(length = grid::unit(8, "pt"), type = "closed")
  ) +
  geom_dag_point(aes(color = role, shape = role), size = 12, alpha = 0.9, stroke = 0) +
  geom_dag_label_repel(
    aes(label = label, fill = role),
    color = "white",
    size = 3.2,
    alpha = 0.75,
    fontface = "bold",
    box.padding = grid::unit(0.4, "lines"),
    point.padding = grid::unit(2, "lines"),
    force = 2,
    max.iter = 5000,
    show.legend = FALSE
  ) +
  scale_color_manual(
    name = "",
    values = role_colors,
    breaks = names(role_colors)
  ) +
  scale_fill_manual(
    name = "",
    values = role_colors,
    breaks = names(role_colors)
  ) +
  scale_shape_manual(
    name = "",
    values = role_shapes,
    breaks = names(role_shapes)
  ) +
  guides(
    color = guide_legend(
      title = "",
      override.aes = list(size = 5, shape = c(19, 19, 17, 15)),
      order = 1
    ),
    fill = "none",
    shape = "none"
  ) +
  theme_dag() +
  expand_plot() +
  labs(
    title = "Manager Training Impact: Causal DAG",
  ) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey40"),
    plot.caption = element_text(size = 10, color = "grey50"),
    legend.position = "right",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 9),
    aspect.ratio = 0.5
  )

# Display the main plot
print(dag_plot)


# Save plot
ggsave("./diagrams/manager_training_dag.png", dag_plot, width = 12, height = 8, dpi = 300)


# ---- Simplified DAG: Performance & Career Focus (confounder structure demo) ----
# 3 nodes only: Performance & Career, Manager Training, Outcomes
# Note: this is a confounder structure (X -> T, X -> Y, T -> Y), not a collider.
# This image is not referenced by the notebook (Slides 7/8 use _detail1/_detail2 PNGs).

# Define node coordinates for simplified 3-node layout (left-to-right)
coord_df_simple <- data.frame(
  name = c("performance_career", "manager_training", "outcomes"),
  x = c(1, 3, 3),
  y = c(0, 0.25, -0.25)
)

# Define the simplified DAG structure
manager_training_dag_simple <- dagify(
  manager_training ~ performance_career,
  outcomes ~ performance_career,
  outcomes ~ manager_training,
  labels = c(
    manager_training = "Manager Training\n(T)",
    outcomes = "Outcomes\n(Efficacy, Workload,\nStay Intention, Retention)\n(Y)",
    performance_career = "Performance\n& Career\n(X)"
  ),
  exposure = "manager_training",
  outcome = "outcomes",
  coords = coord_df_simple
)

# Role mapping for simplified DAG (subset: Treatment, Outcome, Adjusted Confounder)
role_mapping_simple <- c(
  manager_training = "Treatment (Exposure)",
  outcomes = "Outcome",
  performance_career = "Adjusted Confounder"
)
role_colors_simple <- c(
  "Treatment (Exposure)" = "#2E86AB",
  "Outcome" = "#A23B72",
  "Adjusted Confounder" = "#6B7280"
)
role_shapes_simple <- c(
  "Treatment (Exposure)" = 19,
  "Outcome" = 19,
  "Adjusted Confounder" = 15
)

# Create the simplified DAG visualization (same formatting as main DAG)
dag_plot_simple <- manager_training_dag_simple %>%
  tidy_dagitty() %>%
  mutate(role = recode(name, !!!role_mapping_simple)) %>%
  ggplot(aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_dag_edges_link(
    edge_color = "grey60",
    edge_width = 0.8,
    arrow = grid::arrow(length = grid::unit(8, "pt"), type = "closed")
  ) +
  geom_dag_point(aes(color = role, shape = role), size = 12, alpha = 0.9, stroke = 0) +
  geom_dag_label_repel(
    aes(label = label, fill = role),
    color = "white",
    size = 3.2,
    alpha = 0.75,
    fontface = "bold",
    box.padding = grid::unit(0.4, "lines"),
    point.padding = grid::unit(2, "lines"),
    force = 2,
    max.iter = 5000,
    show.legend = FALSE
  ) +
  scale_color_manual(
    name = "",
    values = role_colors_simple,
    breaks = names(role_colors_simple)
  ) +
  scale_fill_manual(
    name = "",
    values = role_colors_simple,
    breaks = names(role_colors_simple)
  ) +
  scale_shape_manual(
    name = "",
    values = role_shapes_simple,
    breaks = names(role_shapes_simple)
  ) +
  guides(
    color = guide_legend(
      title = "",
      override.aes = list(size = 5, shape = c(19, 19, 15)),
      order = 1
    ),
    fill = "none",
    shape = "none"
  ) +
  theme_dag() +
  expand_plot() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "grey40"),
    plot.caption = element_text(size = 10, color = "grey50"),
    legend.position = "right",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 9),
    aspect.ratio = 0.5
  )

# Display the simplified plot
print(dag_plot_simple)

# Save simplified plot
ggsave("./diagrams/dag_confounder_ex.png", dag_plot_simple, width = 12, height = 8, dpi = 300)