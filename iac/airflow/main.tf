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

// The library with methods for creating and
// managing the infrastructure in GCP, this will
// apply to all the resources in the project
provider "google" {
  project = var.project_id
  region  = var.region
}


resource "google_compute_instance" "airflow_vm" {
  name         = var.instance_name
  machine_type = "e2-standard-2"
  zone         = var.zone
  tags         = ["airflow"]

  boot_disk {
    initialize_params {
      image = "ubuntu-2004-focal-v20240307b"
      size  = 50
      type  = "pd-standard" # HDD
    }
  }

  network_interface {
    network = "default"

    access_config {
      # Ephemeral public IP
    }
  }

  service_account {
    email  = google_service_account.airflow_sa.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

}


resource "google_compute_firewall" "airflow_fw" {
  name    = "allow-airflow"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["airflow"]
}


resource "google_service_account" "airflow_sa" {
  account_id   = "airflow-sa"
  display_name = "Service Account for Airflow"
}


resource "google_project_iam_member" "airflow_sa_roles" {
  for_each = toset([
    "roles/storage.admin",
    "roles/bigquery.admin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.airflow_sa.email}"
}

