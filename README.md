# 🚧 Work In Progress

A quick guide to set up and analyze your environment for **Brazil** (`br`):

---

## 🛠️ Setup

```sh
make setup COUNTRY=Brazil CODE=br SEMESTER=202501
```

## ⏱️ Latency Measurement

```sh
make latency COUNTRY=Brazil CODE=br
```

## 🌐 Locality Check (with VPN)

```sh
make locality COUNTRY=Brazil CODE=br VPN=<vpn_name_label>
```

## 📊 Run Analysis

```sh
make analysis COUNTRY=Brazil CODE=br
```

---

## ☁️ Google Cloud Configuration

1. Set your GCP project:

```sh
gcloud config set project <your-project-id>
```

2. Authenticate with Application Default Credentials:

```sh
gcloud auth application-default login
```

---

> **Note:** Replace `<vpn_name_label>` and `<your-project-id>` with your actual VPN label and Google Cloud project ID.
