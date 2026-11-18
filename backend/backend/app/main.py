from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .candidates import NearestArcCandidateProvider, PreparedCandidateProvider
from .config import Settings
from .data import ReferenceRouteRepository, TripRepository
from .hmm import HMMConfig, HMMViterbiDecoder
from .predictor import DistanceFallbackPredictor, JoblibCandidatePredictor, BundleCandidatePredictor
from .schemas import EvaluationResponse, HealthResponse, PredictionResponse, TripSummary
from .services import EvaluationService, PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)


def build_services(settings: Settings):
    trips = TripRepository(settings.data_root, settings.partition)
    production = settings.environment == "production"
    if production and not settings.candidate_dir:
        raise RuntimeError("Production mode requires PATHSNAP_CANDIDATE_DIR from Person 1")
    if settings.candidate_dir:
        if production and not settings.candidate_dir.is_dir():
            raise RuntimeError(f"Prepared candidate directory does not exist: {settings.candidate_dir}")
        candidate_provider = PreparedCandidateProvider(settings.candidate_dir)
    else:
        candidate_provider = NearestArcCandidateProvider(
            radius_m=settings.candidate_radius_m, max_candidates=settings.max_candidates
        )

    if settings.model_mode == "distance_fallback":
        if production or not settings.allow_dev_fallback:
            raise RuntimeError("Distance fallback is development-only; configure Person 2's trained model")
        predictor = DistanceFallbackPredictor(settings.emission_sigma_m)
    elif settings.model_mode == 'bundle':
        if not settings.model_path:
            raise RuntimeError('PATHSNAP_MODEL_PATH must point to a trained model bundle')
        predictor = BundleCandidatePredictor(settings.model_path)
    else:
        if not settings.model_path:
            raise RuntimeError("PATHSNAP_MODEL_PATH must point to Person 2's model artifact")
        predictor = JoblibCandidatePredictor(
            settings.model_path, settings.model_features,
            mode=settings.model_mode, positive_class=settings.positive_class,
        )

    if production:
        if not settings.data_root.is_dir() or not trips.partition_dir.is_dir():
            raise RuntimeError("Production data root or configured partition does not exist")
        trip_ids = trips.list_trip_ids()
        if not trip_ids:
            raise RuntimeError("Production partition contains no valid trip folders")
        available = candidate_provider.available_trip_ids()
        missing_candidate_trips = [trip_id for trip_id in trip_ids if trip_id not in available]
        if missing_candidate_trips:
            raise RuntimeError(
                "Prepared candidate files are missing for trips: "
                + ", ".join(missing_candidate_trips[:10])
            )

    decoder = HMMViterbiDecoder(HMMConfig(
        transition_sigma_base_m=settings.transition_sigma_base_m,
        transition_sigma_fraction=settings.transition_sigma_fraction,
        missing_state_penalty=settings.missing_state_penalty,
        unreachable_penalty=settings.unreachable_penalty,
    ))
    prediction = PredictionService(trips, candidate_provider, predictor, decoder)
    references = ReferenceRouteRepository(trips)
    evaluation = EvaluationService(prediction, references)
    return trips, candidate_provider, predictor, prediction, evaluation


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        trips, candidate_provider, predictor, prediction, evaluation = build_services(settings)
        app.state.trips = trips
        app.state.candidate_provider = candidate_provider
        app.state.predictor = predictor
        app.state.prediction = prediction
        app.state.evaluation = evaluation
        app.state.settings = settings
        logger.info(
            "PathSnap backend ready | env=%s partition=%s predictor=%s mode=%s candidates=%s trips=%d",
            settings.environment, settings.partition, predictor.name, predictor.mode,
            type(candidate_provider).__name__, len(trips.list_trip_ids()),
        )
        yield
        logger.info("PathSnap backend shutting down")

    application = FastAPI(
        title="PathSnap Prediction API",
        description="HMM/Viterbi route reconstruction backend for the Person 3 component.",
        version="1.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_logging(request: Request, call_next):
        response = await call_next(request)
        client = request.client.host if request.client else "unknown"
        logger.info("%s %s -> %s client=%s", request.method, request.url.path, response.status_code, client)
        return response

    def state(request):
        if not hasattr(request.app.state, "prediction"):
            raise HTTPException(status_code=503, detail="Backend services are not initialised")
        return request.app.state

    @application.get("/", tags=["system"])
    def root():
        return {"name": "PathSnap Prediction API", "version": "1.1.0", "docs": "/docs", "health": "/health"}

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        s = state(request)
        is_dev = s.predictor.mode == "development_fallback"
        dataset_available = bool(s.trips.list_trip_ids())
        return HealthResponse(
            status="degraded" if is_dev else ("healthy" if dataset_available else "unavailable"),
            partition=settings.partition,
            model_loaded=not is_dev,
            predictor=s.predictor.name,
            predictor_mode=s.predictor.mode,
            candidate_provider=type(s.candidate_provider).__name__,
            data_root_exists=settings.data_root.exists(),
            dataset_available=dataset_available,
            hmm_enabled=True,
            prediction_mode="development_fallback" if is_dev else "trained_model",
        )

    @application.get("/api/trips", tags=["trips"])
    def list_trips(request: Request):
        s = state(request)
        return {"partition": settings.partition, "trip_ids": s.trips.list_trip_ids()}

    @application.get("/api/trips/{trip_id}", response_model=TripSummary, tags=["trips"])
    def trip_metadata(trip_id: str, request: Request):
        s = state(request)
        try:
            return s.trips.metadata(trip_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/predict/{trip_id}", response_model=PredictionResponse, tags=["prediction"])
    def predict_trip(trip_id: str, request: Request):
        s = state(request)
        try:
            return s.prediction.predict(trip_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Prediction failed for trip %s", trip_id)
            raise HTTPException(status_code=500, detail="Prediction failed; inspect backend logs") from exc

    @application.get("/api/evaluate/{trip_id}", response_model=EvaluationResponse, tags=["evaluation"])
    def evaluate_trip(trip_id: str, request: Request):
        s = state(request)
        try:
            return s.evaluation.evaluate(trip_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Evaluation failed for trip %s", trip_id)
            raise HTTPException(status_code=500, detail="Evaluation failed; inspect backend logs") from exc

    @application.get('/api/maps/{trip_id}', response_class=HTMLResponse, tags=['prediction'])
    def route_map(trip_id: str, request: Request):
        from .maps import render_route_map
        s = state(request)
        try:
            prediction = s.prediction.predict(trip_id)
            return render_route_map(s.trips.load_trip(trip_id), prediction)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return application


app = create_app()
