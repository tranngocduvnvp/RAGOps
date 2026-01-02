# Ref: https://github.com/terraform-google-modules/terraform-google-kubernetes-engine/blob/master/examples/simple_autopilot_public
# To define that we will use GCP
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.80.0" // Provider version
    }
  }
  required_version = "1.14.1" // Terraform version
}


provider "google" {
  project = var.project_id
  region  = var.region
}

// Google Kubernetes Engine
resource "google_container_cluster" "primary" {
  name     = "${var.project_id}-gke"
  location = var.region

  enable_autopilot = false

  initial_node_count = 2

  node_config {
    machine_type = "e2-highmem-2" // 2 vCPUs, 16 GB RAM
    disk_size_gb = 60
  }
}