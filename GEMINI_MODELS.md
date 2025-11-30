# Gemini Model Selection and Fallback

The Custom ML backend uses Google's Gemini API for generating style recommendations. To handle rate limits gracefully, the system implements a **cascading fallback** mechanism.

## How It Works

When you make a recommendation request, the system:

1. **Tries the preferred model** (default: `gemini-2.0-flash-exp`)
2. **If rate limited** → tries `gemini-2.5-flash`
3. **If still rate limited** → tries `gemini-2.0-flash-lite`
4. **If still rate limited** → tries `gemini-flash-latest`
5. **If all models are rate limited** → falls back to basic ML-only recommendations

Each model has its own rate limit pool, so you can often bypass rate limits by trying different models!

## Available Models

| Model | Speed | Quality | Free Tier Limits |
|-------|-------|---------|------------------|
| `gemini-2.0-flash-exp` | Very Fast | Experimental | 10 RPM, 4M TPM |
| `gemini-2.5-flash` | Fast | Best Quality | 15 RPM, 4M TPM |
| `gemini-2.0-flash-lite` | Fastest | Lightweight | 15 RPM, 4M TPM |
| `gemini-flash-latest` | Fast | Auto-Updated | 15 RPM, 4M TPM |

*RPM = Requests Per Minute, TPM = Tokens Per Minute*

## Configuring Your Preferred Model

### Method 1: Environment Variable (Recommended)

Add to your `.env` file:

```bash
# Use gemini-2.5-flash as primary (best quality)
GEMINI_MODEL=gemini-2.5-flash

# Or use lite version for speed
GEMINI_MODEL=gemini-2.0-flash-lite
```

Available options:
- `gemini-2.0-flash-exp` (default, experimental)
- `gemini-2.5-flash` (newest, best quality)
- `gemini-2.0-flash-lite` (fastest, lightweight)
- `gemini-flash-latest` (auto-updated)

### Method 2: Default Behavior

If you don't set `GEMINI_MODEL`, the system uses `gemini-2.0-flash-exp` first, then falls back to others.

## Viewing Which Model Was Used

Check the ML server logs (terminal where you ran `python ml_server.py`):

```
  Trying gemini-2.0-flash-exp...
  ✗ gemini-2.0-flash-exp rate limited, trying next model...
  Trying gemini-2.5-flash...
  ✓ Success with gemini-2.5-flash
```

## Rate Limit Tips

**If you're hitting rate limits frequently:**

1. **Set a lighter model as default:**
   ```bash
   GEMINI_MODEL=gemini-2.0-flash-lite
   ```

2. **Wait between requests:** The free tier resets per minute

3. **Use fewer items:** Fewer items = fewer tokens = lower chance of hitting limits

4. **Upgrade to paid tier:** See [Gemini API Pricing](https://ai.google.dev/pricing)

## Basic Recommendations (All Models Rate Limited)

When all Gemini models are exhausted, you'll get a basic recommendation like:

```
Based on ML analysis for Formal:

Items detected:
1. Female Blue Tshirts - Casual
2. Male Black Jeans - Casual

Suggestion: Consider the formality required for 'Formal' 
when selecting your outfit. Match colors and styles appropriately.
```

This still uses the ViT model's predictions - just without the LLM's detailed styling advice.

## Troubleshooting

**Error: "RESOURCE_EXHAUSTED"**
- All models hit rate limits
- Falls back to basic recommendations automatically
- Wait 60 seconds and try again

**Error: "Invalid model name"**
- Check your `GEMINI_MODEL` value in `.env`
- Must be one of the three supported models

**No styling advice, just basic text**
- Check that `GEMINI_API_KEY` is set correctly in `.env`
- Verify your API key at [Google AI Studio](https://aistudio.google.com/apikey)
