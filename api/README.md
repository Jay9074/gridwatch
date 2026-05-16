# GridWatch Public API

Free, open API for Northeast US power grid outage risk intelligence.

## Live URL
After deploy: `https://gridwatch-api.onrender.com`
Interactive docs: `https://gridwatch-api.onrender.com/docs`

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /` | API metadata |
| `GET /api/v1/health` | Service status |
| `GET /api/v1/storms/active` | All active storm events |
| `GET /api/v1/predictions/active` | All outage predictions |
| `GET /api/v1/predictions/state/{state}` | Predictions for one state |
| `GET /api/v1/predictions/county/{state}/{county}` | One county |
| `GET /api/v1/accuracy` | Public accuracy scorecard |
| `GET /api/v1/counties` | All monitored counties |
| `GET /api/v1/states` | State-level risk summary |
| `GET /docs` | Swagger UI |

## Query Parameters

Most endpoints support filtering:
- `?tier=SEVERE` - filter by storm tier
- `?state=Maine` - filter by state
- `?min_customers=5000` - minimum predicted customer impact
- `?limit=50` - cap results

## Examples

```bash
# Get all severe storms
curl https://gridwatch-api.onrender.com/api/v1/storms/active?tier=SEVERE

# Get predictions for Maine
curl https://gridwatch-api.onrender.com/api/v1/predictions/state/Maine

# Get high-impact predictions only
curl https://gridwatch-api.onrender.com/api/v1/predictions/active?min_customers=10000
```

## Python Example

```python
import requests
api = "https://gridwatch-api.onrender.com/api/v1"

# Active storms
storms = requests.get(f"{api}/storms/active").json()
print(f"{storms['count']} storm events detected")

# State drill-down
nj = requests.get(f"{api}/predictions/state/New Jersey").json()
print(f"NJ: {nj['total_customers_at_risk']:,} customers at risk")
```

## Run Locally

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

## Deploy to Render

1. Push this repo to GitHub
2. Sign in at render.com (free tier)
3. New → Web Service → connect GitHub repo
4. Render detects `render.yaml` and deploys automatically
5. First deploy takes ~3 minutes
6. After deploy, your API is live at `https://gridwatch-api.onrender.com`

## Free Tier Limits

Render free tier:
- Spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds
- 750 hours/month free (enough for 24/7 single service)

For zero spin-down, upgrade to Starter ($7/month) when you're ready.

## License
MIT - free for any use including commercial.
